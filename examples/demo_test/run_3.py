
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


# -----------------------------
# Utilities
# -----------------------------
def set_determinism(seed: int = 1234) -> None:
    # No random numbers used for metrics; we only ensure deterministic behavior.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def json_dump(path: Path, obj: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=False)


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> List[List[int]]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm.tolist()


def macro_f1_from_confusion(cm: np.ndarray) -> float:
    # cm: [C,C], rows true, cols pred
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    # probs: [N,C], y_true: [N]
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    acc = (pred == y_true).astype(np.float64)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (conf > lo) & (conf <= hi) if i > 0 else (conf >= lo) & (conf <= hi)
        if not np.any(m):
            continue
        bin_acc = acc[m].mean()
        bin_conf = conf[m].mean()
        ece += (m.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / np.sum(ex, axis=axis, keepdims=True)


# -----------------------------
# Dataset: AG News -> 5-way deterministic mapping
# -----------------------------
ACTIONS = ["LocalPatch", "TargetedRegenerate", "RefineDecomposition", "ContractUpdate", "WholeRepoRegenerate"]
# AG News labels: 0=World, 1=Sports, 2=Business, 3=Sci/Tech
# Deterministic mapping to 5 actions:
#   World -> LocalPatch
#   Sports -> TargetedRegenerate
#   Business -> RefineDecomposition
#   Sci/Tech -> split deterministically by hash parity into ContractUpdate vs WholeRepoRegenerate
def stable_hash32(s: str) -> int:
    # Simple, deterministic string hash independent of Python hash randomization
    # FNV-1a 32-bit
    h = 2166136261
    for ch in s.encode("utf-8"):
        h ^= ch
        h = (h * 16777619) & 0xFFFFFFFF
    return h


def map_to_action(ag_label: int, text: str) -> int:
    if ag_label == 0:
        return 0  # LocalPatch
    if ag_label == 1:
        return 1  # TargetedRegenerate
    if ag_label == 2:
        return 2  # RefineDecomposition
    if ag_label == 3:
        # Deterministically split Sci/Tech into two action classes
        parity = stable_hash32(text) % 2
        return 3 if parity == 0 else 4
    raise ValueError(f"Unexpected AG label: {ag_label}")


def load_and_prepare_agnews(train_n: int, val_n: int, test_n: int) -> Dict[str, Dict[str, List]]:
    ds = load_dataset("ag_news")

    # Build a pool from train split only for deterministic sampling without randomness:
    # take first K items to avoid random sampling.
    total_needed = train_n + val_n + test_n
    pool = ds["train"].select(range(total_needed))

    texts = []
    y = []
    for ex in pool:
        # Use title + description for richer signal
        txt = (ex["title"] + " " + ex["description"]).strip()
        texts.append(txt)
        y.append(map_to_action(int(ex["label"]), txt))
    y = np.array(y, dtype=np.int64)

    # Stratified split: first train vs temp, then temp into val/test
    # Deterministic: set random_state and shuffle=True; sklearn uses RNG but controlled by random_state.
    idx = np.arange(len(texts))
    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=(val_n + test_n), stratify=y, random_state=1234, shuffle=True
    )
    # Now split temp into val/test with stratification
    val_size = val_n / (val_n + test_n)
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=(1.0 - val_size), stratify=y_temp, random_state=1234, shuffle=True
    )

    # Enforce exact sizes by deterministic truncation (still stratified approximately if overshoots due to rounding)
    train_idx = train_idx[:train_n]
    val_idx = val_idx[:val_n]
    test_idx = test_idx[:test_n]

    def slice_by_idx(idxs: np.ndarray) -> Tuple[List[str], np.ndarray]:
        return [texts[i] for i in idxs], y[idxs]

    train_texts, train_y = slice_by_idx(train_idx)
    val_texts, val_y = slice_by_idx(val_idx)
    test_texts, test_y = slice_by_idx(test_idx)

    return {
        "train": {"texts": train_texts, "y": train_y},
        "val": {"texts": val_texts, "y": val_y},
        "test": {"texts": test_texts, "y": test_y},
    }


def label_distribution(y: np.ndarray, num_classes: int) -> List[int]:
    counts = np.zeros((num_classes,), dtype=np.int64)
    for c in y:
        counts[int(c)] += 1
    return counts.tolist()


# -----------------------------
# Model: TF-IDF + 2-layer MLP
# -----------------------------
class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_p: float):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.drop = nn.Dropout(dropout_p)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = F.relu(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def batch_iter(X: torch.Tensor, y: torch.Tensor, batch_size: int):
    n = X.shape[0]
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        yield X[start:end], y[start:end]


@torch.no_grad()
def eval_model(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> Dict:
    model.eval()
    logits = model(X).cpu().numpy()
    probs = softmax_np(logits, axis=1)
    pred = probs.argmax(axis=1)
    y_np = y.cpu().numpy()
    acc = float((pred == y_np).mean())
    cm = np.array(compute_confusion_matrix(y_np, pred, probs.shape[1]), dtype=np.int64)
    macro_f1 = macro_f1_from_confusion(cm)
    ece10 = expected_calibration_error(probs, y_np, n_bins=10)

    # Cross-entropy loss
    # compute in torch for numeric stability
    logits_t = torch.tensor(logits, dtype=torch.float32)
    y_t = y.cpu()
    loss = float(F.cross_entropy(logits_t, y_t).item())

    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "ece10": ece10,
        "loss": loss,
        "confusion_matrix": cm.tolist(),
        "probs": probs,  # keep for downstream calculations; caller may drop before JSON
        "logits": logits,  # for temperature scaling
        "y_true": y_np,
    }


# -----------------------------
# Temperature scaling (post-hoc calibration)
# -----------------------------
class TemperatureScaler(nn.Module):
    def __init__(self, init_temp: float = 1.0):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(init_temp)], dtype=torch.float32))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        temp = torch.exp(self.log_temp).clamp(min=1e-3, max=1e3)
        return logits / temp

    def temperature(self) -> float:
        return float(torch.exp(self.log_temp).item())


def fit_temperature(val_logits: np.ndarray, val_y: np.ndarray, device: str = "cpu") -> Dict:
    scaler = TemperatureScaler(init_temp=1.0).to(device)
    logits_t = torch.tensor(val_logits, dtype=torch.float32, device=device)
    y_t = torch.tensor(val_y, dtype=torch.long, device=device)

    opt = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=100, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        scaled = scaler(logits_t)
        loss = F.cross_entropy(scaled, y_t)
        loss.backward()
        return loss

    opt.step(closure)

    with torch.no_grad():
        scaled = scaler(logits_t)
        loss = float(F.cross_entropy(scaled, y_t).item())

    return {"temperature": scaler.temperature(), "val_nll_after": loss}


def apply_temperature(logits: np.ndarray, temperature: float) -> np.ndarray:
    return logits / max(temperature, 1e-6)


def downstream_agent_loop_metrics_from_probs(probs: np.ndarray) -> Dict:
    # Proxy from Run 2 definition
    # regeneration_scope = P(TargetedRegenerate) + 2*P(WholeRepoRegenerate)
    regen_scope = probs[:, 1] + 2.0 * probs[:, 4]
    files_touched = 1.0 + 4.5 * regen_scope
    mean_files = float(files_touched.mean())

    # pass@budget: p_success = 0.5 + 0.5*max_prob, budget_iters independent tries
    max_prob = probs.max(axis=1)
    p_success = 0.5 + 0.5 * max_prob
    budget_iters = 30
    p_pass = 1.0 - np.power((1.0 - p_success), budget_iters)
    pass_at_budget_expected = float(p_pass.mean())

    return {
        "pass_at_budget_expected": pass_at_budget_expected,
        "mean_regeneration_scope_proxy_files_touched": mean_files,
        "budget_iters": budget_iters,
        "definition": "Proxy computed from model probabilities: regeneration_scope = P(TargetedRegenerate) + 2*P(WholeRepoRegenerate); files_touched = 1 + 4.5*regen_scope. pass@budget uses per-item p_success = 0.5+0.5*max_prob and assumes budget_iters independent tries.",
    }


def focus_confusion_contract_vs_target(cm: np.ndarray) -> Dict:
    # labels order is ACTIONS
    contract_idx = 3
    targeted_idx = 1
    return {
        "labels": ACTIONS,
        "contractupdate_as_targetedregenerate": int(cm[contract_idx, targeted_idx]),
        "targetedregenerate_as_contractupdate": int(cm[targeted_idx, contract_idx]),
        "contractupdate_support": int(cm[contract_idx, :].sum()),
        "targetedregenerate_support": int(cm[targeted_idx, :].sum()),
    }


# -----------------------------
# Notes: Run 2 writeup payload
# -----------------------------
RUN2_NOTES = r"""
RUN 2 (completed) — Summary for future writeup
=============================================

Run number: 2
Dataset: ag_news (HuggingFace Datasets)
Task framing: "5-way action classification proxy" to emulate an agent decision policy for a verifier-driven
decomposition/localization workflow.

Actions (5 classes, fixed order):
  0 LocalPatch
  1 TargetedRegenerate
  2 RefineDecomposition
  3 ContractUpdate
  4 WholeRepoRegenerate

Deterministic label mapping from AG News:
  - World -> LocalPatch
  - Sports -> TargetedRegenerate
  - Business -> RefineDecomposition
  - Sci/Tech -> split deterministically into (ContractUpdate vs WholeRepoRegenerate) using a stable hash parity of the text.

Motivation for this mapping:
  We need 5 classes but AG News has 4 topics. Splitting Sci/Tech deterministically avoids randomness and creates
  a controlled extra action class while preserving a text signal.

Split protocol and sizes:
  - Total pool: first 9000 samples from ag_news train split.
  - Stratified split into train/val/test with fixed random_state for determinism.
  - Sizes: train=5000, val=2000, test=2000.

Observed class distributions:
  - train: [1250, 1250, 1250, 625, 625]
  - val:   [ 500,  500,  500, 250, 250]
  - test:  [ 500,  500,  500, 250, 250]
This addressed Run 1 issues (perfect accuracy with some classes having zero support), suggesting Run 1 had
label/split leakage or missing classes in splits.

Model:
  - TF-IDF (max_features=1500, ngrams=1..2) -> 2-layer MLP (hidden_dim=64, dropout=0.2) -> 5-way logits
  - Optimizer: AdamW(lr=1e-3, weight_decay=1e-4), epochs=6, CPU
  - Parameter count: 96,389

Training stats (loss):
  - Train loss by epoch: [1.5029, 1.0759, 0.7399, 0.5963, 0.5238, 0.4753]
  - Val loss: 0.5954

Validation metrics:
  - Accuracy: 0.756
  - Macro-F1: 0.65493
  - ECE (10 bins): 0.03472
  - Confusion matrix (rows=true, cols=pred):
    [[430, 29, 26,  2, 13],
     [  9,474, 13,  0,  4],
     [ 24, 15,429,  6, 26],
     [ 16, 13, 32, 28,161],
     [ 17,  7, 43, 32,151]]
  - Focused confusion (ContractUpdate vs TargetedRegenerate):
      contractupdate_as_targetedregenerate: 13 (out of 250 ContractUpdate)
      targetedregenerate_as_contractupdate: 0 (out of 500 TargetedRegenerate)
    Major confusion issue is ContractUpdate vs WholeRepoRegenerate, consistent with being a deterministic split
    of Sci/Tech.

Test metrics:
  - Accuracy: 0.7595
  - Macro-F1: 0.66279
  - ECE (10 bins): 0.02856
  - Confusion matrix:
    [[440, 19, 32,  0,  9],
     [ 16,473,  8,  1,  2],
     [ 24, 10,422,  5, 39],
     [ 29, 12, 33, 34,142],
     [ 24, 12, 30, 34,150]]

Downstream proxy "agent loop" metrics (computed from predicted probabilities):
  - regeneration_scope = P(TargetedRegenerate) + 2*P(WholeRepoRegenerate)
  - files_touched = 1 + 4.5*regen_scope
  - pass@budget: per-item p_success = 0.5 + 0.5*max_prob, assume 30 independent tries
  - Observed: pass_at_budget_expected=1.0, mean_files_touched≈3.293, budget_iters=30

Artifacts:
  - TF-IDF vocab size: 1500

Key takeaways:
  - Removing split/label issues fixed unrealistic perfect accuracy.
  - Model performs reasonably (~0.76 accuracy) but has notable confusion between the two Sci/Tech-derived action
    labels (ContractUpdate vs WholeRepoRegenerate), limiting macro-F1.
  - Calibration error is already low, but downstream loop metrics strongly depend on max probability and thus are
    sensitive to calibration; improving calibration may stabilize agent-policy proxies.

Next planned step (for Run 3):
  Add post-hoc temperature scaling on the validation set to improve calibration (ECE/NLL) while keeping accuracy
  unchanged (argmax invariant under positive temperature). Report metrics before/after calibration and evaluate
  downstream proxy metrics using calibrated probabilities.
""".strip() + "\n"


# -----------------------------
# Main experiment
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    ensure_dir(out_dir)

    # Always update notes.txt in the current working directory (repo root)
    # so another person can find it independent of out_dir.
    notes_path = Path("notes.txt")
    # Append run 2 notes if not already present (idempotent-ish)
    existing = ""
    if notes_path.exists():
        existing = notes_path.read_text(encoding="utf-8")
    if "RUN 2 (completed) — Summary for future writeup" not in existing:
        with notes_path.open("a", encoding="utf-8") as f:
            f.write("\n" + RUN2_NOTES + "\n")

    # Determinism
    set_determinism(1234)
    device = "cpu"

    # Config (matches Run 2)
    train_n, val_n, test_n = 5000, 2000, 2000
    tfidf_max_features = 1500
    ngram_range = (1, 2)
    hidden_dim = 64
    dropout_p = 0.2
    num_classes = 5
    batch_size = 256
    epochs = 6
    lr = 1e-3
    weight_decay = 1e-4

    # Data
    data = load_and_prepare_agnews(train_n=train_n, val_n=val_n, test_n=test_n)
    train_texts, y_train = data["train"]["texts"], data["train"]["y"]
    val_texts, y_val = data["val"]["texts"], data["val"]["y"]
    test_texts, y_test = data["test"]["texts"], data["test"]["y"]

    dists = {
        "train": label_distribution(y_train, num_classes),
        "val": label_distribution(y_val, num_classes),
        "test": label_distribution(y_test, num_classes),
    }

    # Vectorize
    vectorizer = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=ngram_range, lowercase=True)
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    # Convert to torch dense tensors (small enough at 1500 dims)
    X_train_t = torch.tensor(X_train.toarray(), dtype=torch.float32, device=device)
    X_val_t = torch.tensor(X_val.toarray(), dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test.toarray(), dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.long, device=device)
    y_test_t = torch.tensor(y_test, dtype=torch.long, device=device)

    # Model
    model = MLPClassifier(input_dim=X_train_t.shape[1], hidden_dim=hidden_dim, num_classes=num_classes, dropout_p=dropout_p).to(device)
    param_count = count_params(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_loss_by_epoch = []
    for ep in range(epochs):
        model.train()
        # Deterministic order: no shuffling
        losses = []
        for xb, yb in batch_iter(X_train_t, y_train_t, batch_size=batch_size):
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        train_loss_by_epoch.append(float(np.mean(losses)))

    # Eval uncalibrated
    val_eval = eval_model(model, X_val_t, y_val_t)
    test_eval = eval_model(model, X_test_t, y_test_t)

    # Fit temperature on validation logits
    temp_info = fit_temperature(val_eval["logits"], val_eval["y_true"], device=device)
    temperature = temp_info["temperature"]

    # Compute calibrated metrics on val/test
    val_logits_cal = apply_temperature(val_eval["logits"], temperature)
    test_logits_cal = apply_temperature(test_eval["logits"], temperature)
    val_probs_cal = softmax_np(val_logits_cal, axis=1)
    test_probs_cal = softmax_np(test_logits_cal, axis=1)

    # Metrics after calibration
    def metrics_from_probs(probs: np.ndarray, y_true: np.ndarray) -> Dict:
        pred = probs.argmax(axis=1)
        acc = float((pred == y_true).mean())
        cm = np.array(compute_confusion_matrix(y_true, pred, num_classes), dtype=np.int64)
        macro_f1 = macro_f1_from_confusion(cm)
        ece10 = expected_calibration_error(probs, y_true, n_bins=10)
        # NLL
        # add epsilon
        eps = 1e-12
        nll = -float(np.mean(np.log(probs[np.arange(len(y_true)), y_true] + eps)))
        return {"accuracy": acc, "macro_f1": macro_f1, "ece10": ece10, "nll": nll, "confusion_matrix": cm.tolist()}

    val_cal = metrics_from_probs(val_probs_cal, val_eval["y_true"])
    test_cal = metrics_from_probs(test_probs_cal, test_eval["y_true"])

    # Downstream proxy metrics: report both uncalibrated and calibrated using TEST probabilities
    downstream_uncal = downstream_agent_loop_metrics_from_probs(test_eval["probs"])
    downstream_cal = downstream_agent_loop_metrics_from_probs(test_probs_cal)

    # Focus confusion contract vs targeted: before/after (argmax doesn't change with temperature,
    # so confusion should match; we still compute from confusion matrix)
    focus_val = focus_confusion_contract_vs_target(np.array(val_eval["confusion_matrix"], dtype=np.int64))
    focus_test = focus_confusion_contract_vs_target(np.array(test_eval["confusion_matrix"], dtype=np.int64))

    final = {
        "run_number": 3,
        "replanned": False,
        "replan_reason": "Run 2 results are plausible with correct supports; proceed with calibration improvement (temperature scaling) to reduce ECE/NLL and stabilize probability-driven downstream proxy metrics. No need to change dataset/splits/model family.",
        "dataset": "ag_news",
        "task": "5-way action classification proxy (deterministic mapping from AG News topics; one topic split into two actions) for verifier-driven decomposition/localization policy.",
        "actions": ACTIONS,
        "sizes": {"train": train_n, "val": val_n, "test": test_n},
        "label_distributions": dists,
        "model": {
            "type": "TF-IDF + 2-layer MLP",
            "tfidf_max_features": tfidf_max_features,
            "tfidf_ngrams": [ngram_range[0], ngram_range[1]],
            "input_dim": int(X_train_t.shape[1]),
            "hidden_dim": hidden_dim,
            "dropout_p": dropout_p,
            "num_classes": num_classes,
            "param_count": param_count,
            "device": device,
        },
        "train_stats": {
            "train_loss_last": train_loss_by_epoch[-1],
            "train_loss_mean": float(np.mean(train_loss_by_epoch)),
            "train_loss_by_epoch": train_loss_by_epoch,
            "val_loss_uncalibrated": float(val_eval["loss"]),
            "epochs": epochs,
            "optimizer": "AdamW",
            "lr": lr,
            "weight_decay": weight_decay,
            "batch_size": batch_size,
        },
        "val_metrics_uncalibrated": {
            "accuracy": val_eval["accuracy"],
            "macro_f1": val_eval["macro_f1"],
            "ece10": val_eval["ece10"],
            "confusion_matrix": val_eval["confusion_matrix"],
            "confusion_focus_contractupdate_vs_targetedregenerate": focus_val,
        },
        "test_metrics_uncalibrated": {
            "accuracy": test_eval["accuracy"],
            "macro_f1": test_eval["macro_f1"],
            "ece10": test_eval["ece10"],
            "confusion_matrix": test_eval["confusion_matrix"],
            "confusion_focus_contractupdate_vs_targetedregenerate": focus_test,
        },
        "calibration": {
            "method": "temperature_scaling",
            "fit_on": "validation",
            "temperature": temperature,
            "val_nll_after": temp_info["val_nll_after"],
        },
        "val_metrics_calibrated": val_cal,
        "test_metrics_calibrated": test_cal,
        "downstream_agent_loop_metrics_uncalibrated_test": downstream_uncal,
        "downstream_agent_loop_metrics_calibrated_test": downstream_cal,
        "artifacts": {
            "vectorizer_vocab_size": int(len(vectorizer.vocabulary_)),
        },
        "notes": {
            "what_changed_from_run2": "Added post-hoc temperature scaling fit on validation logits; report val/test metrics before and after calibration, plus downstream proxy metrics using calibrated probabilities.",
            "expected_effect": "Accuracy and confusion matrix should remain (nearly) identical (temperature preserves argmax) while ECE and NLL should improve; downstream probability-driven proxies may change.",
        },
    }

    # Remove large arrays before writing JSON
    # (we kept probs/logits in eval outputs for intermediate calculations only)
    json_dump(out_dir / "final_info.json", final)


if __name__ == "__main__":
    main()
