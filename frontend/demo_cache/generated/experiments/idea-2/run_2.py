import argparse
import json
import os
import random
import ast
import difflib
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

try:
    from evaluate import load as load_metric  # pip install evaluate
except Exception:
    load_metric = None

SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


# -----------------------------
# Data
# -----------------------------
def _load_humaneval(max_items: int = 164) -> List[Dict[str, str]]:
    if load_metric is None:
        raise RuntimeError(
            "Please install `evaluate` (pip install evaluate) to load HumanEval."
        )
    ds = load_metric("humaneval")
    items = []
    for i, ex in enumerate(ds["test"][:max_items]):
        prompt = ex.get("prompt", "")
        ref = ex.get("canonical_solution", "")
        if prompt and ref:
            items.append(
                {
                    "task_id": ex.get("task_id", f"HE-{i}"),
                    "prompt": prompt,
                    "reference": ref,
                }
            )
    return items


def load_data():
    """Keep function name. Return train/val/test DataLoaders for training a char-level GRU LM."""
    items = _load_humaneval()
    idx = list(range(len(items)))
    random.Random(SEED).shuffle(idx)
    n = len(items)
    n_tr, n_val = int(0.7 * n), int(0.15 * n)
    train_items = [items[i] for i in idx[:n_tr]]
    val_items = [items[i] for i in idx[n_tr : n_tr + n_val]]
    test_items = [items[i] for i in idx[n_tr + n_val :]]

    # Build vocab on train set (prompts + refs)
    vocab = CharVocab(
        [ex["prompt"] for ex in train_items] + [ex["reference"] for ex in train_items]
    )

    max_len = 1024
    train_ds = HEDataset(train_items, vocab, max_len=max_len)
    val_ds = HEDataset(val_items, vocab, max_len=max_len)
    test_ds = HEDataset(test_items, vocab, max_len=max_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id),
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=8,
        shuffle=False,
        collate_fn=lambda b: collate_fn(b, vocab.pad_id),
    )

    # Return loaders and aux (we keep signature minimal by stashing vocab on dataset)
    return train_loader, val_loader, test_loader


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
    def __init__(self, items: List[Dict[str, str]], vocab: CharVocab, max_len: int):
        self.items = items
        self.vocab = vocab
        self.max_len = max_len
        self.rows = []
        for ex in items:
            ctx = [vocab.bos_id] + vocab.encode(ex["prompt"] + "\n# solution:\n")
            tgt = vocab.encode(ex["reference"].rstrip() + "\n") + [vocab.eos_id]
            x = (ctx + tgt)[:max_len]
            y = x[1:] + [vocab.eos_id]
            ctx_len = min(len(ctx), len(x))
            mask = [0] * ctx_len + [1] * (len(x) - ctx_len)
            self.rows.append((x, y, mask, ex["reference"], ex["task_id"]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx]


def collate_fn(batch, pad_id: int):
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


# -----------------------------
# Model (keep class name SingleLayerGRU)
# -----------------------------
class SingleLayerGRU(nn.Module):
    def __init__(self, vocab_size: int, emb: int = 128, hidden: int = 256):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb, padding_idx=0)
        self.rnn = nn.GRU(emb, hidden, num_layers=1, batch_first=True)
        self.head = nn.Linear(hidden, vocab_size)

    def forward(self, x):
        e = self.emb(x)
        h, _ = self.rnn(e)
        return self.head(h)  # [B, L, V]


# -----------------------------
# Train / Evaluate
# -----------------------------
def _masked_ce(logits, targets, mask, pad_id: int):
    B, L, V = logits.shape
    loss = nn.functional.cross_entropy(
        logits.reshape(B * L, V),
        targets.reshape(B * L),
        ignore_index=pad_id,
        reduction="none",
    ).reshape(B, L)
    loss = (loss * mask).sum() / (mask.sum() + 1e-8)
    return loss


def train(model, train_loader, optimizer, criterion, device):
    """Keep function name; internally we use masked CE over target region."""
    model.train()
    for epoch in range(3):
        tot = 0.0
        steps = 0
        for X, Y, M, _, _ in train_loader:
            X, Y, M = X.to(device), Y.to(device), M.to(device)
            optimizer.zero_grad()
            logits = model(X)
            loss = _masked_ce(logits, Y, M, pad_id=0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tot += loss.item()
            steps += 1
        print(f"Epoch {epoch+1}: train_loss={tot/max(1,steps):.4f}")
    print(f"Training completed over {epoch+1} epochs.")


@torch.no_grad()
def _greedy_generate(
    model, vocab: CharVocab, prompt: str, device: str = "cpu", max_new_tokens: int = 512
) -> str:
    model.eval()
    ctx = [vocab.bos_id] + vocab.encode(prompt + "\n# solution:\n")
    x = torch.tensor([ctx], dtype=torch.long, device=device)
    out = ctx.copy()
    for _ in range(max_new_tokens):
        logits = model(x)[:, -1, :]
        nid = int(torch.argmax(logits, dim=-1).item())
        out.append(nid)
        x = torch.tensor([out], dtype=torch.long, device=device)
        if nid == vocab.eos_id:
            break
    gen = out[len(ctx) :]
    return vocab.decode(gen)


def _ast_ok(code: str) -> bool:
    try:
        ast.parse(code)
        return True
    except SyntaxError:
        return False


def _undef_refs(code: str) -> int:
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


def _text_sim(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def _pass1_proxy(gen: str, ref: str) -> int:
    return int(gen.strip() == ref.strip())


def evaluate(model, data_loader, device):
    """Keep function name; now returns coherence-oriented metrics on the test set."""
    # retrieve vocab from dataset via closure trick
    ds = data_loader.dataset
    vocab = ds.vocab  # type: ignore
    items = data_loader.dataset.items  # type: ignore

    ast_cnt = 0
    undef_sum = 0
    sim_sum = 0.0
    pass_sum = 0
    n = 0
    gens = []

    for _, _, _, refs, tids in data_loader:
        for ref, tid in zip(refs, tids):
            prompt = next(ex["prompt"] for ex in items if ex["task_id"] == tid)
            gen = _greedy_generate(
                model, vocab, prompt, device=device, max_new_tokens=512
            )
            gens.append({"task_id": tid, "generated": gen, "reference": ref})
            ast_cnt += int(_ast_ok(gen))
            undef_sum += _undef_refs(gen)
            sim_sum += _text_sim(gen, ref)
            pass_sum += _pass1_proxy(gen, ref)
            n += 1

    metrics = {
        "AST_Parse_Rate": ast_cnt / max(1, n),
        "UndefinedRef_Avg": undef_sum / max(1, n),
        "TextSim_Avg": sim_sum / max(1, n),
        "pass@1_proxy": pass_sum / max(1, n),
    }
    return metrics, gens


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data()

    # Build model from dataset vocab
    vocab = train_loader.dataset.vocab  # type: ignore
    model = SingleLayerGRU(vocab_size=len(vocab.itos), emb=128, hidden=256).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion kept for signature compatibility, not used directly (masked CE inside train())
    criterion = nn.CrossEntropyLoss()

    train(model, train_loader, optimizer, criterion, device)

    # Optional: quick val loop (omitted to keep minimal changes)

    # Evaluate on test
    test_metrics, gens = evaluate(model, test_loader, device)

    with open(os.path.join(out_dir, "final_info.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(out_dir, "generations.jsonl"), "w") as f:
        for g in gens:
            f.write(json.dumps(g) + "\n")

    print(json.dumps(test_metrics, indent=2))
    print("Experiment completed successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    args = parser.parse_args()
    main(args.out_dir)
