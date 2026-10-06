"""ProsQA data: tokenizer, sample formatting, collation.

A graph question is serialised as

    <eos> s1 t1 | s2 t2 | ... [Q] c1 c2 [R] root [A] target <eos>

where (s, t) are the directed edges (shuffled each time the sample is drawn),
c1/c2 the two candidate answers in random order, and `target` the answer.
Everything except the `target` position is clean conditioning (`cond_mask` = 1);
the target position is the single slot the flow has to fill.
"""
from __future__ import annotations

import gzip
import json
import random
from collections import defaultdict, deque

import torch
from torch.utils.data import Dataset

NUM_NODE_TOKENS = 31
SPECIAL_TOKENS = ["<|start-latent|>", "<|end-latent|>", "<|latent|>", "|", "[Q]", "[R]", "[A]",
                  "<eos>", "<|no-answer|>"]   # ids 31..39; the first three and the last are unused
                                              # but kept so that vocab_size = 40


class Tokenizer:
    """Whitespace tokenizer over node ids "0".."30" and the special tokens (40 ids)."""

    def __init__(self):
        self.vocab = {str(i): i for i in range(NUM_NODE_TOKENS)}
        for i, tok in enumerate(SPECIAL_TOKENS, start=NUM_NODE_TOKENS):
            self.vocab[tok] = i
        self.pad_id = self.vocab["<eos>"]

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def encode(self, text: str) -> list[int]:
        return [self.vocab[tok] for tok in text.split()]


def shortest_path(edges, root: int, target: int) -> list[int]:
    """BFS shortest path root -> target as a list of node ids (= token ids)."""
    adj = defaultdict(list)
    for s, t in edges:
        adj[int(s)].append(int(t))
    root, target = int(root), int(target)
    prev = {root: -1}
    queue = deque([root])
    while queue and target not in prev:
        u = queue.popleft()
        for v in adj[u]:
            if v not in prev:
                prev[v] = u
                queue.append(v)
    assert target in prev, "ProsQA targets are reachable by construction"
    path, node = [], target
    while node != -1:
        path.append(node)
        node = prev[node]
    return path[::-1]


def format_sample(sample: dict, tok: Tokenizer) -> dict:
    edges = list(sample["edges"])
    random.shuffle(edges)
    candidates = [sample["target"], sample["neg_target"]]
    random.shuffle(candidates)
    question = ("<eos> " + " | ".join(f"{s} {t}" for s, t in edges)
                + f" [Q] {candidates[0]} {candidates[1]} [R] {sample['root']}")
    q_ids = tok.encode(question)
    c_ids = tok.encode(f"[A] {sample['target']} <eos>")
    input_ids = q_ids + c_ids
    answer_idx = len(q_ids) + 1                       # position of `target`
    cond_mask = [1] * len(input_ids)
    cond_mask[answer_idx] = 0                         # the only noised position
    path = shortest_path(sample["edges"], sample["root"], sample["target"])
    return {
        "input_ids": input_ids,
        "cond_mask": cond_mask,
        "attention_mask": [1] * len(input_ids),
        "answer_idx": answer_idx,
        "answer_id": int(sample["target"]),
        "path": path,                                 # [root, ..., target]
        "path_len": len(path) - 1,                    # number of hops
    }


class ProsQADataset(Dataset):
    def __init__(self, path: str, tok: Tokenizer):
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt") as f:
            self.raw = json.load(f)
        self.tok = tok

    def __len__(self) -> int:
        return len(self.raw)

    def __getitem__(self, idx: int) -> dict:
        return format_sample(self.raw[idx], self.tok)


class Collator:
    """Right-pad to the longest sequence of the batch. Padding is clean conditioning
    (cond_mask = 1) and masked out of attention; paths are padded with their target."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, items: list[dict]) -> dict:
        L = max(len(it["input_ids"]) for it in items)
        P = max(len(it["path"]) for it in items)
        pad = lambda seq, val, n: seq + [val] * (n - len(seq))
        return {
            "input_ids": torch.tensor([pad(it["input_ids"], self.pad_id, L) for it in items]),
            "cond_mask": torch.tensor([pad(it["cond_mask"], 1, L) for it in items], dtype=torch.bool),
            "attention_mask": torch.tensor([pad(it["attention_mask"], 0, L) for it in items], dtype=torch.bool),
            "answer_idx": torch.tensor([it["answer_idx"] for it in items]),
            "answer_id": torch.tensor([it["answer_id"] for it in items]),
            "path": torch.tensor([pad(it["path"], it["path"][-1], P) for it in items]),
            "path_len": torch.tensor([it["path_len"] for it in items]),
        }
