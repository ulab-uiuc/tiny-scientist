#!/usr/bin/env python3
"""
Lightweight HumanEval experiment for idea-3:
- Train a single-layer GRU language model on prompt→solution pairs.
- Report BLEU and ROUGE-L on the held-out test split.
"""

import argparse
import json
import math
import os
import random
from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from datasets import load_dataset  # pip install datasets
except Exception:
    load_dataset = None

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# Data loading / preprocessing
# ---------------------------------------------------------------------------
def load_humaneval(max_items: int = 164) -> List[Dict[str, str]]:
    if load_dataset is None:
        raise RuntimeError(
            "Please install `datasets` (pip install datasets) to load HumanEval."
        )
    dataset = load_dataset("openai_humaneval")
    split_name = "test" if "test" in dataset else next(iter(dataset.keys()))
    records = dataset[split_name]
    tasks: List[Dict[str, str]] = []
    for i, ex in enumerate(records):
        if i >= max_items:
            break
        prompt = ex.get("prompt", "")
        reference = ex.get("canonical_solution", "")
        if prompt and reference:
            tasks.append(
                {
                    "task_id": ex.get("task_id", f"HE-{i}"),
                    "prompt": prompt,
                    "reference": reference,
                }
            )
    if not tasks:
        raise RuntimeError("HumanEval dataset yielded no usable tasks.")
    return tasks


def pseudo_split(items: Sequence[Dict[str, str]]) -> Tuple[List, List, List]:
    idx = list(range(len(items)))
    random.Random(SEED).shuffle(idx)
    n = len(items)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train = [items[i] for i in idx[:n_train]]
    val = [items[i] for i in idx[n_train : n_train + n_val]]
    test = [items[i] for i in idx[n_train + n_val :]]
    return train, val, test


class CharVocab:
    def __init__(self, texts: Sequence[str]):
        specials = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"]
        charset = set()
        for t in texts:
            charset.update(t)
        self.itos = specials + sorted(ch for ch in charset if ch not in specials)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.unk_id = self.stoi["<UNK>"]

    def encode(self, text: str) -> List[int]:
        return [self.stoi.get(ch, self.unk_id) for ch in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))


class HEDataset(Dataset):
    def __init__(
        self,
        items: Sequence[Dict[str, str]],
        vocab: CharVocab,
        max_len: int = 512,
    ):
        self.vocab = vocab
        self.rows: List[Tuple[List[int], List[int], List[int], str, str, str]] = []
        for ex in items:
            prompt = ex["prompt"]
            ref = ex["reference"].rstrip() + "\n"
            ctx = [vocab.bos_id] + vocab.encode(prompt + "\n# solution:\n")
            tgt = vocab.encode(ref) + [vocab.eos_id]
            x = (ctx + tgt)[:max_len]
            y = x[1:] + [vocab.eos_id]
            ctx_len = min(len(ctx), len(x))
            mask = [0] * ctx_len + [1] * (len(x) - ctx_len)
            self.rows.append((x, y, mask, ref, ex["task_id"], prompt))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        return self.rows[idx]


def collate_fn(batch, pad_id: int):
    length = max(len(x) for x, _, _, _, _, _ in batch)
    xs, ys, masks, refs, tids, prompts = [], [], [], [], [], []
    for x, y, m, ref, tid, prompt in batch:
        pad = length - len(x)
        xs.append(x + [pad_id] * pad)
        ys.append(y + [pad_id] * pad)
        masks.append(m + [0] * pad)
        refs.append(ref)
        tids.append(tid)
        prompts.append(prompt)
    return (
        torch.tensor(xs, dtype=torch.long),
        torch.tensor(ys, dtype=torch.long),
        torch.tensor(masks, dtype=torch.float32),
        refs,
        tids,
        prompts,
    )


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class SingleLayerGRU(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 256, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.emb(x)
        h, _ = self.rnn(emb)
        return self.head(h)


def masked_ce(logits, targets, mask, pad_id: int) -> torch.Tensor:
    B, L, V = logits.shape
    loss = nn.functional.cross_entropy(
        logits.reshape(B * L, V),
        targets.reshape(B * L),
        ignore_index=pad_id,
        reduction="none",
    ).reshape(B, L)
    return (loss * mask).sum() / (mask.sum() + 1e-8)


@torch.no_grad()
def greedy_decode(
    model: nn.Module,
    vocab: CharVocab,
    prompt: str,
    max_new: int,
    device: torch.device,
) -> str:
    model.eval()
    ctx = [vocab.bos_id] + vocab.encode(prompt + "\n# solution:\n")
    x = torch.tensor([ctx], dtype=torch.long, device=device)
    out = ctx.copy()
    for _ in range(max_new):
        logits = model(x)[:, -1, :]
        nid = int(torch.argmax(logits, dim=-1).item())
        out.append(nid)
        x = torch.tensor([out], dtype=torch.long, device=device)
        if nid == vocab.eos_id:
            break
    return vocab.decode(out[len(ctx) :])


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def _ngram_counts(tokens: Sequence[str], max_order: int) -> Counter:
    counts: Counter = Counter()
    for order in range(1, max_order + 1):
        for i in range(len(tokens) - order + 1):
            counts[tuple(tokens[i : i + order])] += 1
    return counts


def compute_bleu(reference: str, candidate: str, max_order: int = 4) -> float:
    ref_tokens = reference.strip().split()
    cand_tokens = candidate.strip().split()
    if not cand_tokens:
        return 0.0
    ref_counts = _ngram_counts(ref_tokens, max_order)
    cand_counts = _ngram_counts(cand_tokens, max_order)
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    for ngram, count in cand_counts.items():
        matches_by_order[len(ngram) - 1] += min(count, ref_counts.get(ngram, 0))
    for order in range(1, max_order + 1):
        possible_matches_by_order[order - 1] = max(
            0, len(cand_tokens) - order + 1
        )
    precisions = []
    for match, possible in zip(matches_by_order, possible_matches_by_order):
        precisions.append((match + 1) / (possible + 1))
    geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    if cand_len == 0:
        return 0.0
    ratio = cand_len / (ref_len + 1e-8)
    brevity = 1.0 if ratio > 1.0 else math.exp(1.0 - 1.0 / (ratio + 1e-8))
    return float(geo_mean * brevity)


def lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token in a:
        curr = [0] * (len(b) + 1)
        for j, tok_b in enumerate(b, start=1):
            if token == tok_b:
                curr[j] = prev[j - 1] + 1
            else:
                curr[j] = max(prev[j], curr[j - 1])
        prev = curr
    return prev[-1]


def compute_rouge_l(reference: str, candidate: str) -> float:
    ref_tokens = reference.strip().split()
    cand_tokens = candidate.strip().split()
    if not ref_tokens or not cand_tokens:
        return 0.0
    lcs = lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return (2 * precision * recall) / (precision + recall)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------
def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    pad_id: int,
    epochs: int,
) -> None:
    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        steps = 0
        for X, Y, M, *_ in train_loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = masked_ce(logits, Y, M, pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item())
            steps += 1
        print(f"[Epoch {epoch}] train_loss={total / max(1, steps):.4f}")


@torch.no_grad()
def run_eval(
    model: nn.Module,
    loader: DataLoader,
    vocab: CharVocab,
    device: torch.device,
    decode_max: int,
) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    model.eval()
    bleu_sum = 0.0
    rouge_sum = 0.0
    gens: List[Dict[str, str]] = []
    count = 0
    for _, _, _, refs, tids, prompts in loader:
        for ref, tid, prompt in zip(refs, tids, prompts):
            gen = greedy_decode(model, vocab, prompt, decode_max, device)
            gens.append({"task_id": tid, "generated": gen, "reference": ref})
            bleu_sum += compute_bleu(ref, gen)
            rouge_sum += compute_rouge_l(ref, gen)
            count += 1
    metrics = {
        "BLEU": bleu_sum / max(1, count),
        "ROUGE_L": rouge_sum / max(1, count),
        "Samples": count,
    }
    return metrics, gens


def main(
    out_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    emb: int,
    hidden: int,
    max_len: int,
    decode_max: int,
    max_items: int,
):
    os.makedirs(out_dir, exist_ok=True)
    items = load_humaneval(max_items=max_items)
    train_items, val_items, test_items = pseudo_split(items)

    vocab = CharVocab(
        [ex["prompt"] for ex in train_items] + [ex["reference"] for ex in train_items]
    )
    train_ds = HEDataset(train_items, vocab, max_len=max_len)
    val_ds = HEDataset(val_items, vocab, max_len=max_len)
    test_ds = HEDataset(test_items, vocab, max_len=max_len)

    coll = lambda batch: collate_fn(batch, vocab.pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SingleLayerGRU(len(vocab.itos), emb=emb, hidden=hidden).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    train_model(model, train_loader, optimizer, device, vocab.pad_id, epochs)

    # quick validation monitoring
    with torch.no_grad():
        val_metrics, _ = run_eval(model, val_loader, vocab, device, decode_max)
        print(f"[Validation] BLEU={val_metrics['BLEU']:.4f} ROUGE_L={val_metrics['ROUGE_L']:.4f}")

    test_metrics, gens = run_eval(model, test_loader, vocab, device, decode_max)
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(out_dir, "generations.jsonl"), "w") as f:
        for row in gens:
            f.write(json.dumps(row) + "\n")
    print(json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--emb", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--decode_max", type=int, default=512)
    parser.add_argument("--max_items", type=int, default=164)
    args = parser.parse_args()
    main(
        args.out_dir,
        args.epochs,
        args.batch_size,
        args.lr,
        args.emb,
        args.hidden,
        args.max_len,
        args.decode_max,
        args.max_items,
    )
