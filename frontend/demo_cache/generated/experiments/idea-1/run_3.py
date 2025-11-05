#!/usr/bin/env python3
# experiment.py — HumanEval prompt→solution char-GRU training (keeps "training" intact)
# Outputs: final_info.json (metrics) + generations.jsonl
#
# Usage:
#   python experiment.py --out_dir runs/idea1 --epochs 3 --batch_size 8
#
# Notes:
# - We use a deterministic pseudo-split (70/15/15) from HumanEval test-only items to preserve train/val/test structure.
# - For true pass@k, plug your unit-test runner where indicated.

import argparse
import json
import os
import random
import ast
import difflib
from typing import List, Dict, Tuple

try:
    from evaluate import load as load_metric  # pip install evaluate
except Exception:
    load_metric = None

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


def load_humaneval(max_items: int = 164) -> List[Dict[str, str]]:
    if load_metric is None:
        raise RuntimeError(
            "Please install `evaluate` (pip install evaluate) to load HumanEval."
        )
    ds = load_metric("humaneval")
    tasks = []
    for i, ex in enumerate(ds["test"][:max_items]):
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
    return tasks


def pseudo_split(items: List[Dict[str, str]]) -> Tuple[List, List, List]:
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
    def __init__(self, texts: List[str]):
        specials = ["<PAD>", "<BOS>", "<EOS>", "<SEP>"]
        charset = set()
        for t in texts:
            charset.update(t)
        self.itos = specials + sorted(ch for ch in charset if ch not in specials)
        self.stoi = {ch: i for i, ch in enumerate(self.itos)}
        self.pad_id = self.stoi["<PAD>"]
        self.bos_id = self.stoi["<BOS>"]
        self.eos_id = self.stoi["<EOS>"]
        self.sep_id = self.stoi["<SEP>"]

    def encode(self, s: str) -> List[int]:
        return [self.stoi.get(ch, self.sep_id) for ch in s]

    def decode(self, ids: List[int]) -> str:
        return "".join(self.itos[i] for i in ids if 0 <= i < len(self.itos))


class HEDataset(Dataset):
    def __init__(
        self, pairs: List[Dict[str, str]], vocab: CharVocab, max_len: int = 1024
    ):
        self.vocab = vocab
        self.max_len = max_len
        self.data = []
        for ex in pairs:
            ctx_ids = [vocab.bos_id] + vocab.encode(ex["prompt"] + "\n# solution:\n")
            tgt_ids = vocab.encode(ex["reference"].rstrip() + "\n") + [vocab.eos_id]
            x = (ctx_ids + tgt_ids)[:max_len]
            ctx_len = min(len(ctx_ids), len(x))
            y = x[1:] + [vocab.eos_id]
            mask = [0] * ctx_len + [1] * (len(x) - ctx_len)
            self.data.append((x, y, mask, ex["reference"], ex["task_id"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate(batch, pad_id: int):
    L = max(len(x) for x, _, _, _, _ in batch)
    X = []
    Y = []
    M = []
    refs = []
    tids = []
    for x, y, m, ref, tid in batch:
        pad = L - len(x)
        X.append(x + [pad_id] * pad)
        Y.append(y + [pad_id] * pad)
        M.append(m + [0] * pad)
        refs.append(ref)
        tids.append(tid)
    return (
        torch.tensor(X, dtype=torch.long),
        torch.tensor(Y, dtype=torch.long),
        torch.tensor(M, dtype=torch.float32),
        refs,
        tids,
    )


class CharGRU(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 128, hidden: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.rnn(e)
        return self.head(h)


def masked_ce(logits, targets, mask, pad_id: int):
    B, L, V = logits.shape
    loss = nn.functional.cross_entropy(
        logits.reshape(B * L, V),
        targets.reshape(B * L),
        ignore_index=pad_id,
        reduction="none",
    ).reshape(B, L)
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return loss


@torch.no_grad()
def greedy_decode(
    model, vocab: CharVocab, prompt: str, max_new: int = 512, device: str = "cpu"
):
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
    gen = out[len(ctx) :]
    return vocab.decode(gen)


def ast_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def undef_refs(code: str) -> int:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 999
    defined, used = set(), set()

    class V(ast.NodeVisitor):
        def visit_FunctionDef(self, node):
            defined.add(node.name)
            self.generic_visit(node)

        def visit_ClassDef(self, node):
            defined.add(node.name)
            self.generic_visit(node)

        def visit_Assign(self, node):
            for t in node.targets:
                if hasattr(t, "id"):
                    defined.add(t.id)
            self.generic_visit(node)

        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Load):
                used.add(node.id)

    V().visit(tree)
    ignore = set(dir(__builtins__)) | {"True", "False", "None", "self"}
    unresolved = [n for n in used if n not in defined and n not in ignore]
    return len(unresolved)


def text_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def pass1_proxy(gen: str, ref: str) -> int:
    return int(gen.strip() == ref.strip())


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
    train, val, test = pseudo_split(items)

    vocab = CharVocab(
        [ex["prompt"] for ex in train] + [ex["reference"] for ex in train]
    )
    train_ds = HEDataset(train, vocab, max_len=max_len)
    val_ds = HEDataset(val, vocab, max_len=max_len)
    test_ds = HEDataset(test, vocab, max_len=max_len)

    coll = lambda b: collate(b, vocab.pad_id)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, collate_fn=coll
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, collate_fn=coll
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, collate_fn=coll
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = CharGRU(vocab_size=len(vocab.itos), emb=emb, hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best = 1e9
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        steps = 0
        for X, Y, M, _, _ in train_loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            opt.zero_grad()
            logits = model(X)
            loss = masked_ce(logits, Y, M, vocab.pad_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
            steps += 1
        tr_loss = tot / max(1, steps)

        model.eval()
        vtot = 0.0
        vsteps = 0
        with torch.no_grad():
            for X, Y, M, _, _ in val_loader:
                X, Y, M = X.to(device), Y.to(device), M.to(device)
                vtot += masked_ce(model(X), Y, M, vocab.pad_id).item()
                vsteps += 1
        val_loss = vtot / max(1, vsteps)
        print(f"[Epoch {ep}] train_loss={tr_loss:.4f} val_loss={val_loss:.4f}")
        if val_loss < best:
            best = val_loss
            torch.save(model.state_dict(), os.path.join(out_dir, "best.pt"))

    # Reload best and evaluate on test with greedy decode
    if os.path.exists(os.path.join(out_dir, "best.pt")):
        model.load_state_dict(
            torch.load(os.path.join(out_dir, "best.pt"), map_location=device)
        )

    gens = []
    ast_cnt = 0
    undef_sum = 0
    sim_sum = 0.0
    pass_cnt = 0
    for X, Y, M, refs, tids in test_loader:
        for ref, tid in zip(refs, tids):
            gen = greedy_decode(
                model,
                vocab,
                (
                    items[0]["prompt"]
                    if False
                    else next(ex["prompt"] for ex in test if ex["task_id"] == tid)
                ),
                max_new=decode_max,
                device=device,
            )
            gens.append({"task_id": tid, "generated": gen, "reference": ref})
            ast_cnt += int(ast_ok(gen))
            undef_sum += undef_refs(gen)
            sim_sum += text_sim(gen, ref)
            pass_cnt += pass1_proxy(gen, ref)

    n = len(test)
    results = {
        "Dataset": f"HumanEval pseudo-split (n_train={len(train)}, n_val={len(val)}, n_test={len(test)})",
        "Test": {
            "AST_Parse_Rate": ast_cnt / max(1, n),
            "UndefinedRef_Avg": undef_sum / max(1, n),
            "TextSim_Avg": sim_sum / max(1, n),
            "pass@1_proxy": pass_cnt / max(1, n),
        },
        "Notes": "Plug a unit-test runner for true pass@k.",
    }
    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(out_dir, "generations.jsonl"), "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--emb", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--decode_max", type=int, default=512)
    p.add_argument("--max_items", type=int, default=164)
    args = p.parse_args()
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
