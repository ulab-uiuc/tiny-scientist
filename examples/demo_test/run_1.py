
import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score


# -----------------------------
# Utilities: deterministic text processing and label construction
# -----------------------------
_ACTIONS = [
    "LocalPatch",
    "TargetedRegenerate",
    "RefineDecomposition",
    "ContractUpdate",
    "WholeRepoRegenerate",
]
_ACTION_TO_ID = {a: i for i, a in enumerate(_ACTIONS)}


def _strip_punct_lower(text: str) -> str:
    """
    Lightweight preprocessing: lowercase + strip punctuation.
    This keeps the experiment cheap while approximating "verifier output text" normalization.
    """
    text = text.lower()
    # Replace punctuation with space (keep alphanumerics/underscore)
    text = re.sub(r"[^a-z0-9_\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _iteration_bucket(i: int) -> str:
    # Deterministic bucketization (no randomness)
    if i <= 2:
        return "iter_0_2"
    if i <= 5:
        return "iter_3_5"
    if i <= 10:
        return "iter_6_10"
    return "iter_11_plus"


def _files_touched_bucket(n: int) -> str:
    if n <= 2:
        return "files_0_2"
    if n <= 5:
        return "files_3_5"
    if n <= 10:
        return "files_6_10"
    return "files_11_plus"


def _simulate_agent_context(ex_idx: int, base_action: str) -> Tuple[str, str, str]:
    """
    Deterministically generate (last_action, iteration_bucket, files_touched_bucket)
    using the example index and base_action. This avoids randomness while providing
    controlled, repeatable variation in the input string.
    """
    last_action = _ACTIONS[ex_idx % len(_ACTIONS)]
    # tie iteration to index to create distribution across buckets
    iteration_index = (ex_idx % 12) + 1
    files_touched = (ex_idx % 13)
    return last_action, _iteration_bucket(iteration_index), _files_touched_bucket(files_touched)


def _inject_verifier_tokens(text: str, latent_label: int) -> str:
    """
    Deterministically inject "verifier-like" tokens to emulate typed failure families.
    This is a controlled corruption that remains lightweight and reproducible.

    latent_label: original AG News label in {0,1,2,3}
      0: World -> contract/schema/type drift
      1: Sports -> assertion/local patch
      2: Business -> import/dependency / targeted regeneration
      3: Sci/Tech -> decomposition/circular deps
    """
    # Minimal, consistent injection: prefix + suffix.
    if latent_label == 0:
        tag = "signaturemismatch schematypemismatch contractlocked"
    elif latent_label == 1:
        tag = "assertionfail localpatch stacktrace"
    elif latent_label == 2:
        tag = "importerror missingsymbol targetedregenerate"
    else:
        tag = "circulardependency refinedecomposition oscillation"
    return f"{tag} {text} {tag}"


def _map_to_action_id(processed_text: str, latent_label: int) -> int:
    """
    Map AG News 4-class label into 5-way action classes with severity override:
      - World(0) -> ContractUpdate
      - Sports(1) -> LocalPatch
      - Business(2) -> TargetedRegenerate
      - Sci/Tech(3) -> RefineDecomposition
      - Extra class: if text length > 180 tokens -> WholeRepoRegenerate override

    Token length is computed after preprocessing and injection.
    """
    tok_len = len(processed_text.split())
    if tok_len > 180:
        return _ACTION_TO_ID["WholeRepoRegenerate"]

    if latent_label == 0:
        return _ACTION_TO_ID["ContractUpdate"]
    if latent_label == 1:
        return _ACTION_TO_ID["LocalPatch"]
    if latent_label == 2:
        return _ACTION_TO_ID["TargetedRegenerate"]
    return _ACTION_TO_ID["RefineDecomposition"]


# -----------------------------
# Dataset loading + featurization
# -----------------------------
@dataclass
class DataBundle:
    X_train: torch.Tensor
    y_train: torch.Tensor
    X_val: torch.Tensor
    y_val: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    vectorizer: TfidfVectorizer


def _stratified_select_indices(labels: List[int], n: int) -> List[int]:
    """
    Deterministic stratified subsampling: take the first k indices per class,
    where k is proportional to class frequency, then distribute remainders.
    No randomness is used. This ensures sizes ≤ (5000/2000/2000) as required.
    """
    labels = np.asarray(labels)
    classes, counts = np.unique(labels, return_counts=True)
    total = counts.sum()
    # initial allocation by floor of proportional counts
    alloc = {c: int(math.floor(n * (cnt / total))) for c, cnt in zip(classes, counts)}
    # adjust to exact n by distributing leftover deterministically by class id
    current = sum(alloc.values())
    remainder = n - current
    # add 1 to smallest class ids first (deterministic)
    for c in sorted(classes.tolist()):
        if remainder <= 0:
            break
        alloc[c] += 1
        remainder -= 1

    selected = []
    for c in sorted(classes.tolist()):
        idxs = np.where(labels == c)[0].tolist()
        selected.extend(idxs[: alloc[c]])
    # If due to extreme imbalance we didn't reach n, pad deterministically
    if len(selected) < n:
        for i in range(len(labels)):
            if i not in set(selected):
                selected.append(i)
            if len(selected) == n:
                break
    return selected[:n]


def load_data() -> DataBundle:
    """
    Loads AG News and constructs train/val/test with maximum sizes:
      train=5000, val=2000, test=2000.

    Uses TF-IDF with max_features=1500, unigrams+bigrams, L2 normalization
    (TfidfVectorizer defaults to L2 norm). This keeps the experiment lightweight:
    sparse bag-of-words features avoid expensive tokenization/LLMs.
    """
    ds = load_dataset("ag_news")

    train_texts = ds["train"]["text"]
    train_labels = ds["train"]["label"]
    test_texts = ds["test"]["text"]
    test_labels = ds["test"]["label"]

    # Deterministic, stratified subsampling as specified
    train_idx = _stratified_select_indices(train_labels, n=5000)

    # From original test split: sample 4000 stratified, then split into 2000/2000
    test_4k_idx = _stratified_select_indices(test_labels, n=4000)
    val_idx = test_4k_idx[:2000]
    test_idx = test_4k_idx[2000:4000]

    def build_inputs(texts: List[str], labels: List[int], indices: List[int]) -> Tuple[List[str], np.ndarray]:
        out_texts = []
        out_y = []
        for j, i in enumerate(indices):
            raw = texts[i]
            latent = int(labels[i])

            normalized = _strip_punct_lower(raw)
            verifierish = _inject_verifier_tokens(normalized, latent)

            # Deterministic agent context features
            base_action = "NA"
            last_action, iter_b, files_b = _simulate_agent_context(ex_idx=i, base_action=base_action)

            # Concatenated string: [verifier_output_text] + [last_action] + [iteration_bucket] + [num_files_touched_bucket]
            combined = f"{verifierish} last_action={last_action} {iter_b} {files_b}"

            action_id = _map_to_action_id(combined, latent)
            out_texts.append(combined)
            out_y.append(action_id)
        return out_texts, np.asarray(out_y, dtype=np.int64)

    tr_texts, y_tr = build_inputs(train_texts, train_labels, train_idx)
    va_texts, y_va = build_inputs(test_texts, test_labels, val_idx)
    te_texts, y_te = build_inputs(test_texts, test_labels, test_idx)

    # TF-IDF: word unigrams+bigrams, max_features=1500, lowercase already done.
    vectorizer = TfidfVectorizer(
        max_features=1500,
        ngram_range=(1, 2),
        lowercase=False,
        norm="l2",
    )
    X_tr = vectorizer.fit_transform(tr_texts)
    X_va = vectorizer.transform(va_texts)
    X_te = vectorizer.transform(te_texts)

    # Convert to dense float32 tensors (1500 dims is small enough at this scale).
    X_tr_t = torch.tensor(X_tr.toarray(), dtype=torch.float32)
    X_va_t = torch.tensor(X_va.toarray(), dtype=torch.float32)
    X_te_t = torch.tensor(X_te.toarray(), dtype=torch.float32)

    y_tr_t = torch.tensor(y_tr, dtype=torch.long)
    y_va_t = torch.tensor(y_va, dtype=torch.long)
    y_te_t = torch.tensor(y_te, dtype=torch.long)

    return DataBundle(
        X_train=X_tr_t,
        y_train=y_tr_t,
        X_val=X_va_t,
        y_val=y_va_t,
        X_test=X_te_t,
        y_test=y_te_t,
        vectorizer=vectorizer,
    )


# -----------------------------
# Model
# -----------------------------
class ShallowMLP(nn.Module):
    def __init__(self, input_dim: int = 1500, hidden_dim: int = 64, num_classes: int = 5, dropout_p: float = 0.2):
        super().__init__()
        # Architecture: Linear(1500->64) + ReLU + Dropout(0.2) + Linear(64->5)
        # Parameter count: (1500*64+64) + (64*5+5) = 96,064 + 325 = 96,389 parameters (under ~100k)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model() -> nn.Module:
    """
    Constructs the lightweight 2-layer MLP specified in the plan.
    This is the simplest non-linear classifier that can still learn interactions
    between TF-IDF features while staying under ~100k parameters.
    """
    return ShallowMLP(input_dim=1500, hidden_dim=64, num_classes=5, dropout_p=0.2)


# -----------------------------
# Training / evaluation
# -----------------------------
def train(model: nn.Module, train_loader: torch.utils.data.DataLoader, optimizer, criterion, device: torch.device, epochs: int = 5) -> Dict[str, float]:
    """
    Standard supervised training loop with gradient updates.
    Kept to <=5 epochs to ensure fast execution on CPU while allowing convergence on 5k examples.
    """
    model.train()
    losses = []
    for epoch in range(1, epochs + 1):
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu().item()))

        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else float("nan")
        losses.append(mean_loss)
        print(f"epoch={epoch} train_loss={mean_loss:.4f}")
    return {"train_loss_last": losses[-1], "train_loss_mean": float(np.mean(losses))}


def _expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Expected Calibration Error (ECE) with equally spaced bins on max predicted probability.
    """
    confidences = probs.max(axis=1)
    preds = probs.argmax(axis=1)
    accuracies = (preds == y_true).astype(np.float32)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for b0, b1 in zip(bins[:-1], bins[1:]):
        mask = (confidences > b0) & (confidences <= b1) if b1 < 1.0 else (confidences > b0) & (confidences <= b1 + 1e-8)
        if not np.any(mask):
            continue
        bin_acc = float(accuracies[mask].mean())
        bin_conf = float(confidences[mask].mean())
        ece += float(mask.mean()) * abs(bin_acc - bin_conf)
    return float(ece)


def evaluate(model: nn.Module, data_loader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, object]:
    """
    Computes metrics from real predictions:
    - Accuracy and Macro-F1 match the primary goal: overall action-selection quality under class imbalance.
    - Confusion matrix slice emphasizes ContractUpdate vs TargetedRegenerate, common drift/localization confusion.
    - ECE measures calibration to support fallback decisions (e.g., WholeRepoRegenerate on low confidence).
    """
    model.eval()
    all_logits = []
    all_y = []
    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            logits = model(xb).cpu()
            all_logits.append(logits)
            all_y.append(yb.clone())

    logits = torch.cat(all_logits, dim=0).numpy()
    y_true = torch.cat(all_y, dim=0).numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro"))

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(_ACTIONS))))
    cu = _ACTION_TO_ID["ContractUpdate"]
    tr = _ACTION_TO_ID["TargetedRegenerate"]
    confusion_focus = {
        "labels": _ACTIONS,
        "contractupdate_as_targetedregenerate": int(cm[cu, tr]),
        "targetedregenerate_as_contractupdate": int(cm[tr, cu]),
        "contractupdate_support": int(cm[cu, :].sum()),
        "targetedregenerate_support": int(cm[tr, :].sum()),
    }

    ece10 = _expected_calibration_error(probs=probs, y_true=y_true, n_bins=10)

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ece10": float(ece10),
        "confusion_matrix": cm.tolist(),
        "confusion_focus_contractupdate_vs_targetedregenerate": confusion_focus,
    }


def _simulate_pass_at_budget(y_true: np.ndarray, y_pred: np.ndarray, budget_iters: int = 30) -> Dict[str, float]:
    """
    Deterministic simulated verifier loop metric:
    - Each true action corresponds to a dominant "failure family".
    - Predicted action determines:
        (a) scope cost (files touched proxy)
        (b) probability of resolving that failure type within an iteration
    We then compute expected pass probability within a fixed iteration budget
    using a geometric model: P(pass within B) = 1 - (1 - p) ** B.

    This uses no randomness; it is an analytic expectation based on model outputs.
    """

    # Files-touched scope proxy per action (smaller is better)
    scope_cost = {
        _ACTION_TO_ID["LocalPatch"]: 1.0,
        _ACTION_TO_ID["TargetedRegenerate"]: 4.0,
        _ACTION_TO_ID["RefineDecomposition"]: 7.0,
        _ACTION_TO_ID["ContractUpdate"]: 6.0,
        _ACTION_TO_ID["WholeRepoRegenerate"]: 12.0,
    }

    # Resolution probability matrix p(true_action, predicted_action)
    # Higher on-diagonal; reasonable off-diagonal asymmetries:
    # - WholeRepoRegenerate has decent chance across types but high scope cost.
    # - ContractUpdate vs TargetedRegenerate confusion is penalized.
    base = np.full((5, 5), 0.10, dtype=np.float64)
    np.fill_diagonal(base, 0.35)
    whole = _ACTION_TO_ID["WholeRepoRegenerate"]
    base[:, whole] = 0.25

    lp = _ACTION_TO_ID["LocalPatch"]
    tr = _ACTION_TO_ID["TargetedRegenerate"]
    rd = _ACTION_TO_ID["RefineDecomposition"]
    cu = _ACTION_TO_ID["ContractUpdate"]

    # Specialize a few relations to match narrative
    base[cu, tr] = 0.12
    base[tr, cu] = 0.12
    base[rd, tr] = 0.14
    base[tr, rd] = 0.14
    base[lp, tr] = 0.15
    base[tr, lp] = 0.15
    base[cu, cu] = 0.40
    base[tr, tr] = 0.40
    base[lp, lp] = 0.38
    base[rd, rd] = 0.36

    probs = base[y_true, y_pred]
    pass_probs = 1.0 - np.power((1.0 - probs), budget_iters)
    pass_at_budget = float(pass_probs.mean())

    mean_scope = float(np.mean([scope_cost[int(a)] for a in y_pred]))
    return {
        "pass_at_budget_expected": pass_at_budget,
        "mean_regeneration_scope_proxy_files_touched": mean_scope,
        "budget_iters": int(budget_iters),
    }


def _make_loaders(bundle: DataBundle, batch_size: int = 64) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    train_ds = torch.utils.data.TensorDataset(bundle.X_train, bundle.y_train)
    val_ds = torch.utils.data.TensorDataset(bundle.X_val, bundle.y_val)
    test_ds = torch.utils.data.TensorDataset(bundle.X_test, bundle.y_test)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


def main(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    bundle = load_data()
    print(f"dataset_sizes: train={bundle.X_train.shape[0]} val={bundle.X_val.shape[0]} test={bundle.X_test.shape[0]}")
    print(f"feature_dim={bundle.X_train.shape[1]} (TF-IDF max_features=1500)")

    train_loader, val_loader, test_loader = _make_loaders(bundle, batch_size=64)

    model = build_model().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    train_stats = train(model, train_loader, optimizer, criterion, device, epochs=5)

    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    # Simulated downstream metrics computed from real model predictions on test set
    # (uses deterministic expected value, no randomness)
    model.eval()
    with torch.no_grad():
        logits = model(bundle.X_test.to(device)).cpu().numpy()
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    y_pred = probs.argmax(axis=1)
    y_true = bundle.y_test.numpy()
    downstream = _simulate_pass_at_budget(y_true=y_true, y_pred=y_pred, budget_iters=30)

    final = {
        "dataset": "ag_news",
        "task": "5-way action classification proxy for verifier-driven decomposition/localization policy",
        "actions": _ACTIONS,
        "model": {
            "type": "TF-IDF + 2-layer MLP",
            "tfidf_max_features": 1500,
            "tfidf_ngrams": [1, 2],
            "input_dim": 1500,
            "hidden_dim": 64,
            "dropout_p": 0.2,
            "num_classes": 5,
            "param_count_comment": "Approx 96,389 parameters: (1500*64+64) + (64*5+5).",
        },
        "sizes": {
            "train": int(bundle.X_train.shape[0]),
            "val": int(bundle.X_val.shape[0]),
            "test": int(bundle.X_test.shape[0]),
        },
        "train_stats": train_stats,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "downstream_agent_loop_metrics": downstream,
    }

    out_path = os.path.join(out_dir, "final_info.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.out_dir)
