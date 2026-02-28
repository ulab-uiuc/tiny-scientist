
import argparse
import json
import math
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# ----------------------------
# Utilities: metrics & logging
# ----------------------------

def softmax_np(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> float:
    """
    Standard ECE with equally spaced confidence bins in [0,1].
    probs: (N, C) probabilities
    y_true: (N,) int labels
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == y_true).astype(np.float32)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == 0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences > lo) & (confidences <= hi)
        if not np.any(mask):
            continue
        bin_acc = float(np.mean(accuracies[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def write_notes_run1(notes_path: Path) -> None:
    """
    Include all relevant information for writeup on Run 1.
    """
    run1 = {
        'dataset': 'ag_news',
        'task': '5-way action classification proxy for verifier-driven decomposition/localization policy',
        'actions': ['LocalPatch', 'TargetedRegenerate', 'RefineDecomposition', 'ContractUpdate', 'WholeRepoRegenerate'],
        'model': {
            'type': 'TF-IDF + 2-layer MLP',
            'tfidf_max_features': 1500,
            'tfidf_ngrams': [1, 2],
            'input_dim': 1500,
            'hidden_dim': 64,
            'dropout_p': 0.2,
            'num_classes': 5,
            'param_count_comment': 'Approx 96,389 parameters: (1500*64+64) + (64*5+5).'
        },
        'sizes': {'train': 5000, 'val': 2000, 'test': 2000},
        'train_stats': {'train_loss_last': 0.023860071962581404, 'train_loss_mean': 0.3526673573811975},
        'val_metrics': {
            'accuracy': 1.0,
            'macro_f1': 1.0,
            'ece10': 0.018795825123786927,
            'confusion_matrix': [[1000, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 1000, 0],
                                 [0, 0, 0, 0, 0]],
            'confusion_focus_contractupdate_vs_targetedregenerate': {
                'labels': ['LocalPatch', 'TargetedRegenerate', 'RefineDecomposition', 'ContractUpdate', 'WholeRepoRegenerate'],
                'contractupdate_as_targetedregenerate': 0,
                'targetedregenerate_as_contractupdate': 0,
                'contractupdate_support': 1000,
                'targetedregenerate_support': 0
            }
        },
        'test_metrics': {
            'accuracy': 1.0,
            'macro_f1': 1.0,
            'ece10': 0.017078947722911835,
            'confusion_matrix': [[0, 0, 0, 0, 0],
                                 [0, 1000, 0, 0, 0],
                                 [0, 0, 1000, 0, 0],
                                 [0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0]],
            'confusion_focus_contractupdate_vs_targetedregenerate': {
                'labels': ['LocalPatch', 'TargetedRegenerate', 'RefineDecomposition', 'ContractUpdate', 'WholeRepoRegenerate'],
                'contractupdate_as_targetedregenerate': 0,
                'targetedregenerate_as_contractupdate': 0,
                'contractupdate_support': 0,
                'targetedregenerate_support': 1000
            }
        },
        'downstream_agent_loop_metrics': {
            'pass_at_budget_expected': 0.9999991232152697,
            'mean_regeneration_scope_proxy_files_touched': 5.5,
            'budget_iters': 30
        }
    }

    analysis = (
        "RUN 1 WRITEUP NOTES (verbatim)\n"
        "Run number: 1\n"
        "Experiment description:\n"
        "- Goal: Build a 5-way 'action' classifier as a proxy for a verifier-driven software agent policy.\n"
        "- Dataset: AG News (4-class news topic classification dataset). A mapping or transformation was used to create 5 action labels:\n"
        "  [LocalPatch, TargetedRegenerate, RefineDecomposition, ContractUpdate, WholeRepoRegenerate].\n"
        "- Model: TF-IDF (max_features=1500, ngrams=(1,2)) feeding a small 2-layer MLP (hidden_dim=64, dropout=0.2) with 5 outputs.\n"
        "- Data sizes: train=5000, val=2000, test=2000.\n"
        "Run 1 reported results (from prior run):\n"
        f"{json.dumps(run1, indent=2)}\n\n"
        "Important interpretation / concerns:\n"
        "- The reported val/test confusion matrices indicate some classes have *zero support* in a split.\n"
        "  For example, in val: TargetedRegenerate_support=0; in test: ContractUpdate_support=0.\n"
        "- Achieving 100% accuracy/macro-F1 with missing classes strongly suggests a pipeline bug:\n"
        "  likely label mapping error (e.g., collapsing to fewer than 5 classes), incorrect split construction,\n"
        "  or leakage / data duplication between splits.\n"
        "- Therefore the next experiment (Run 2) will re-plan to enforce:\n"
        "  (1) deterministic, explicit mapping from AG News labels to 5 action labels,\n"
        "  (2) stratified splitting to ensure all 5 actions appear in train/val/test,\n"
        "  (3) explicit sanity checks on label distribution per split,\n"
        "  (4) metrics computed from real model execution.\n\n"
    )

    notes_path.parent.mkdir(parents=True, exist_ok=True)
    with notes_path.open("a", encoding="utf-8") as f:
        f.write(analysis)
        f.write("\n" + ("-" * 80) + "\n\n")


# ----------------------------
# Model
# ----------------------------

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_classes: int, dropout_p: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)


def count_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# ----------------------------
# Data: create 5-way proxy task
# ----------------------------

ACTIONS = ["LocalPatch", "TargetedRegenerate", "RefineDecomposition", "ContractUpdate", "WholeRepoRegenerate"]

def map_agnews_to_action(y_ag: int, idx_within_class: int) -> int:
    """
    Deterministic 5-way proxy mapping.
    AG News labels are 0..3.
    We create 5 actions by splitting one AG label into two actions deterministically.
    - label 0 -> LocalPatch
    - label 1 -> TargetedRegenerate
    - label 2 -> RefineDecomposition
    - label 3 -> split into ContractUpdate and WholeRepoRegenerate based on parity of idx_within_class
    """
    if y_ag == 0:
        return 0
    if y_ag == 1:
        return 1
    if y_ag == 2:
        return 2
    if y_ag == 3:
        return 3 if (idx_within_class % 2 == 0) else 4
    raise ValueError(f"Unexpected AG label: {y_ag}")


def prepare_splits(train_size: int, val_size: int, test_size: int, seed: int = 0):
    ds = load_dataset("ag_news")

    # Use the official train split as pool; we'll create our own train/val/test.
    texts = ds["train"]["text"]
    labels = ds["train"]["label"]

    # Build deterministic per-class indices so the split of label 3 is stable and doesn't depend on global ordering.
    per_class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    action_labels = []
    for y in labels:
        idx = per_class_counts[y]
        per_class_counts[y] += 1
        action_labels.append(map_agnews_to_action(y, idx))
    action_labels = np.array(action_labels, dtype=np.int64)

    # First, take a stratified subset of the pool to match requested total size.
    total = train_size + val_size + test_size
    indices = np.arange(len(texts))

    idx_sub, _ = train_test_split(
        indices,
        train_size=total,
        random_state=seed,
        shuffle=True,
        stratify=action_labels
    )

    texts_sub = [texts[i] for i in idx_sub]
    y_sub = action_labels[idx_sub]

    # Split into train / temp, then temp into val/test, each stratified.
    idx_train, idx_temp = train_test_split(
        np.arange(total),
        train_size=train_size,
        random_state=seed,
        shuffle=True,
        stratify=y_sub
    )

    y_temp = y_sub[idx_temp]
    temp_size = val_size + test_size

    # fraction for val within temp
    val_frac = val_size / temp_size
    idx_val_rel, idx_test_rel = train_test_split(
        np.arange(temp_size),
        train_size=val_frac,
        random_state=seed,
        shuffle=True,
        stratify=y_temp
    )

    idx_val = idx_temp[idx_val_rel]
    idx_test = idx_temp[idx_test_rel]

    def take(idxs):
        X = [texts_sub[i] for i in idxs]
        y = y_sub[idxs]
        return X, y

    X_train, y_train = take(idx_train)
    X_val, y_val = take(idx_val)
    X_test, y_test = take(idx_test)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def label_distribution(y: np.ndarray, num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for v in y.tolist():
        counts[int(v)] += 1
    return counts


# ----------------------------
# Training & evaluation
# ----------------------------

def make_loaders(X_train_vec, y_train, X_val_vec, y_val, X_test_vec, y_test, batch_size: int = 64):
    def to_tensor_dataset(X, y):
        X_t = torch.from_numpy(X.astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.int64))
        return TensorDataset(X_t, y_t)

    train_ds = to_tensor_dataset(X_train_vec, y_train)
    val_ds = to_tensor_dataset(X_val_vec, y_val)
    test_ds = to_tensor_dataset(X_test_vec, y_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader


@torch.no_grad()
def predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_y = []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb).detach().cpu().numpy()
        all_logits.append(logits)
        all_y.append(yb.numpy())
    logits = np.concatenate(all_logits, axis=0)
    y = np.concatenate(all_y, axis=0)
    probs = softmax_np(logits, axis=1)
    return probs, y


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 6,
    lr: float = 1e-3,
    weight_decay: float = 1e-4
) -> Dict[str, Any]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    for epoch in range(epochs):
        model.train()
        running = 0.0
        n = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.detach().cpu().item()) * xb.size(0)
            n += xb.size(0)
        train_losses.append(running / max(1, n))

    # quick val loss
    model.eval()
    val_running, val_n = 0.0, 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_running += float(loss.detach().cpu().item()) * xb.size(0)
            val_n += xb.size(0)
    val_loss = val_running / max(1, val_n)

    return {
        "train_loss_last": float(train_losses[-1]),
        "train_loss_mean": float(np.mean(train_losses)),
        "train_loss_by_epoch": [float(x) for x in train_losses],
        "val_loss": float(val_loss),
        "epochs": int(epochs),
        "optimizer": "AdamW",
        "lr": float(lr),
        "weight_decay": float(weight_decay),
    }


def eval_metrics(probs: np.ndarray, y_true: np.ndarray, labels: List[str]) -> Dict[str, Any]:
    y_pred = np.argmax(probs, axis=1)
    acc = float(accuracy_score(y_true, y_pred))
    macro = float(f1_score(y_true, y_pred, average="macro"))
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(labels)))).tolist()
    ece10 = float(expected_calibration_error(probs, y_true, n_bins=10))

    # Focus confusion for ContractUpdate (3) vs TargetedRegenerate (1)
    cu = 3
    tr = 1
    cu_as_tr = int(np.sum((y_true == cu) & (y_pred == tr)))
    tr_as_cu = int(np.sum((y_true == tr) & (y_pred == cu)))
    cu_support = int(np.sum(y_true == cu))
    tr_support = int(np.sum(y_true == tr))

    return {
        "accuracy": acc,
        "macro_f1": macro,
        "ece10": ece10,
        "confusion_matrix": cm,
        "confusion_focus_contractupdate_vs_targetedregenerate": {
            "labels": labels,
            "contractupdate_as_targetedregenerate": cu_as_tr,
            "targetedregenerate_as_contractupdate": tr_as_cu,
            "contractupdate_support": cu_support,
            "targetedregenerate_support": tr_support,
        }
    }


def downstream_agent_loop_proxy(test_probs: np.ndarray, budget_iters: int = 30) -> Dict[str, Any]:
    """
    A simple *measured* proxy: interpret predicted prob of 'TargetedRegenerate' and
    'WholeRepoRegenerate' as "regeneration scope" propensity.
    Then convert to a deterministic expected "files touched" and pass@budget.
    This is NOT ground-truth agent eval, but is computed from real model outputs.
    """
    # Indices: TargetedRegenerate=1, WholeRepoRegenerate=4
    regen_scope = test_probs[:, 1] + 2.0 * test_probs[:, 4]
    mean_scope = float(np.mean(1.0 + 4.5 * regen_scope))  # between 1 and ~10
    # Expected pass@budget: higher confidence in any class -> higher expected pass.
    confidence = np.max(test_probs, axis=1)
    # Deterministic transformation (not random): probability of success per item
    p_success = np.clip(0.5 + 0.5 * confidence, 0.0, 1.0)
    # Expected probability of succeeding within budget_iters independent attempts (proxy)
    # p_fail_all = Π (1 - p_success)^{budget_iters} approximated by mean item success.
    expected_pass = float(np.mean(1.0 - np.power(1.0 - p_success, budget_iters)))
    return {
        "pass_at_budget_expected": expected_pass,
        "mean_regeneration_scope_proxy_files_touched": mean_scope,
        "budget_iters": int(budget_iters),
        "definition": (
            "Proxy computed from model probabilities: regeneration_scope = P(TargetedRegenerate) + 2*P(WholeRepoRegenerate); "
            "files_touched = 1 + 4.5*regen_scope. "
            "pass@budget uses per-item p_success = 0.5+0.5*max_prob and assumes budget_iters independent tries."
        )
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Update notes for Run 1 (for future writeup)
    write_notes_run1(out_dir / "notes.txt")

    # ----------------------------
    # Run 2: fixed pipeline
    # ----------------------------
    run_number = 2
    seed = 0
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    sizes = {"train": 5000, "val": 2000, "test": 2000}
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_splits(
        train_size=sizes["train"],
        val_size=sizes["val"],
        test_size=sizes["test"],
        seed=seed
    )

    # Sanity checks: ensure all classes appear in each split
    dist_train = label_distribution(y_train, len(ACTIONS))
    dist_val = label_distribution(y_val, len(ACTIONS))
    dist_test = label_distribution(y_test, len(ACTIONS))

    # TF-IDF
    tfidf_max_features = 1500
    tfidf = TfidfVectorizer(max_features=tfidf_max_features, ngram_range=(1, 2), lowercase=True)
    X_train_vec = tfidf.fit_transform(X_train).toarray()
    X_val_vec = tfidf.transform(X_val).toarray()
    X_test_vec = tfidf.transform(X_test).toarray()

    # Model
    input_dim = X_train_vec.shape[1]
    hidden_dim = 64
    dropout_p = 0.2
    num_classes = len(ACTIONS)

    model = MLPClassifier(input_dim=input_dim, hidden_dim=hidden_dim, num_classes=num_classes, dropout_p=dropout_p)
    param_count = count_params(model)

    # Train/eval
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, test_loader = make_loaders(
        X_train_vec, y_train,
        X_val_vec, y_val,
        X_test_vec, y_test,
        batch_size=64
    )

    train_stats = train_model(model, train_loader, val_loader, device, epochs=6, lr=1e-3, weight_decay=1e-4)

    val_probs, val_y = predict_proba(model, val_loader, device)
    test_probs, test_y = predict_proba(model, test_loader, device)

    val_metrics = eval_metrics(val_probs, val_y, ACTIONS)
    test_metrics = eval_metrics(test_probs, test_y, ACTIONS)
    agent_metrics = downstream_agent_loop_proxy(test_probs, budget_iters=30)

    final = {
        "run_number": run_number,
        "replanned": True,
        "replan_reason": (
            "Run 1 showed perfect accuracy with missing classes in splits (zero support for some actions), "
            "suggesting a split/label-mapping issue or leakage. Run 2 enforces deterministic 5-way mapping "
            "and stratified splits with distribution logging."
        ),
        "dataset": "ag_news",
        "task": (
            "5-way action classification proxy (deterministic mapping from AG News topics; one topic split into two actions) "
            "for verifier-driven decomposition/localization policy."
        ),
        "actions": ACTIONS,
        "sizes": sizes,
        "label_distributions": {
            "train": dist_train,
            "val": dist_val,
            "test": dist_test
        },
        "model": {
            "type": "TF-IDF + 2-layer MLP",
            "tfidf_max_features": tfidf_max_features,
            "tfidf_ngrams": [1, 2],
            "input_dim": int(input_dim),
            "hidden_dim": int(hidden_dim),
            "dropout_p": float(dropout_p),
            "num_classes": int(num_classes),
            "param_count": int(param_count),
            "device": str(device),
        },
        "train_stats": train_stats,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "downstream_agent_loop_metrics": agent_metrics,
        "artifacts": {
            "vectorizer_vocab_size": int(len(tfidf.vocabulary_)),
        }
    }

    with (out_dir / "final_info.json").open("w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)

    # Also append Run 2 summary to notes.txt for writeup continuity
    with (out_dir / "notes.txt").open("a", encoding="utf-8") as f:
        f.write("RUN 2 PLAN AND IMPLEMENTATION\n")
        f.write(f"Run number: {run_number}\n")
        f.write("Change from Run 1:\n")
        f.write("- Replanned due to suspiciously perfect metrics and missing-class splits.\n")
        f.write("- Deterministic 5-way mapping from AG News labels (split original label=3 into two actions by per-class parity).\n")
        f.write("- Stratified subset + stratified train/val/test splits to preserve 5-way class presence.\n")
        f.write("- Added label distribution logging and computed metrics from executed model outputs.\n")
        f.write("Key hyperparameters:\n")
        f.write(f"- TF-IDF: max_features={tfidf_max_features}, ngram_range=(1,2)\n")
        f.write(f"- MLP: input_dim={input_dim}, hidden_dim={hidden_dim}, dropout_p={dropout_p}, num_classes={num_classes}\n")
        f.write(f"- Train: epochs={train_stats['epochs']}, lr={train_stats['lr']}, weight_decay={train_stats['weight_decay']}\n")
        f.write(f"- Params: {param_count}\n")
        f.write("Results are saved in final_info.json\n")
        f.write("\n" + ("-" * 80) + "\n\n")


if __name__ == "__main__":
    main()
