"""Train a latent flow on ProsQA and evaluate answer accuracy vs. number of Euler steps.

    python train.py configs/fixed_tN.yaml

Writes to <save_dir>/<name>/: best.pt (best validation accuracy at `eval_steps`),
best_N_<n>.pt (best validation accuracy at every evaluated n), history.json
(per-epoch loss, per-evaluation accuracies) and results.json (final N-sweep of
best.pt on validation and test).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Collator, ProsQADataset, Tokenizer
from flow import LatentFlow
from model import DiT


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build(cfg: dict, vocab_size: int, device):
    model = DiT(vocab_size=vocab_size, max_len=cfg["max_len"], d_model=cfg["d_model"],
                n_layers=cfg["n_layers"], n_heads=cfg["n_heads"], mlp_ratio=cfg["mlp_ratio"],
                dropout=cfg["dropout"], time_embed_layers=cfg["time_embed_layers"],
                freq_dim=cfg["freq_dim"], time_dim=cfg["time_dim"]).to(device)
    flow = LatentFlow(vocab_size=vocab_size, rollout_steps=cfg["rollout_steps"],
                      grad_decay=cfg["grad_decay"], grid=cfg["grid"], t0_max=cfg["t0_max"],
                      t_max=cfg["t_max"], fixed_tN=cfg["fixed_tN"], random_tN=cfg["random_tN"],
                      t_clamp_eps=cfg["t_clamp_eps"], label_smoothing=cfg["label_smoothing"],
                      bf16_acc32=cfg["bf16_acc32"], max_stage=cfg["max_stage"],
                      replay_prob=cfg["replay_prob"])
    return model, flow


@torch.no_grad()
def evaluate(model, flow, loader, device, num_steps: int, pad_id: int) -> float:
    """Fraction of questions whose decoded answer slot equals the target.
    The answer slot is overwritten with the pad id before sampling, so the model
    only ever sees the graph and the question."""
    model.eval()
    correct = total = 0
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["input_ids"] = batch["input_ids"].masked_fill(~batch["cond_mask"], pad_id)
        pred = flow.sample(model, batch, num_steps)
        pred_at_answer = pred.gather(1, batch["answer_idx"].unsqueeze(1)).squeeze(1)
        correct += (pred_at_answer == batch["answer_id"]).sum().item()
        total += pred_at_answer.numel()
    return correct / max(total, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    print("config:", cfg)
    set_seed(cfg["seed"])
    assert torch.cuda.is_available(), "a GPU is required"
    device = "cuda"

    tok = Tokenizer()
    collate = Collator(tok.pad_id)
    loaders = {}
    for split in ("train", "valid", "test"):
        ds = ProsQADataset(os.path.join(cfg["data_dir"], f"prosqa_{split}.json"), tok)
        loaders[split] = DataLoader(ds, batch_size=cfg["batch_size"], shuffle=(split == "train"),
                                    drop_last=(split == "train"), collate_fn=collate,
                                    num_workers=cfg["num_workers"])
    model, flow = build(cfg, tok.vocab_size, device)
    print(f"model: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")

    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    ema_decay = cfg["ema_decay"]

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"],
                                  betas=tuple(cfg["betas"]))
    warmup = cfg["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: (step + 1) / warmup if step < warmup else 1.0)

    save_dir = os.path.join(cfg["save_dir"], cfg["name"])
    os.makedirs(save_dir, exist_ok=True)
    eval_Ns = list(cfg["eval_steps_during_training"])
    primary = cfg["eval_steps"]
    history = {"config": cfg, "epochs": [], "evals": []}
    best = {n: -1.0 for n in eval_Ns}
    best_epoch = {n: -1 for n in eval_Ns}

    def save(path, epoch, val_acc):
        torch.save({"model": model.state_dict(), "ema_model": ema.state_dict(),
                    "epoch": epoch, "val_acc": val_acc, "config": cfg}, path)

    for epoch in range(cfg["num_epochs"]):
        flow.stage = min(epoch // cfg["epochs_per_stage"], cfg["max_stage"])
        model.train()
        running, n = 0.0, 0
        pbar = tqdm(loaders["train"], desc=f"epoch {epoch:03d} [stage {flow.stage}/{cfg['max_stage']}]")
        for batch in pbar:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            loss = flow.loss(model, batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                for p_ema, p in zip(ema.parameters(), model.parameters()):
                    p_ema.mul_(ema_decay).add_(p, alpha=1.0 - ema_decay)
            running += loss.item()
            n += 1
            pbar.set_postfix(loss=f"{running / n:.4f}")
        history["epochs"].append({"epoch": epoch, "train_loss": running / max(n, 1), "stage": flow.stage})

        if (epoch + 1) % cfg["eval_every"] == 0:
            # validation only: the test set is evaluated once, at the end, on the selected checkpoint
            val = {n: evaluate(ema, flow, loaders["valid"], device, n, tok.pad_id) for n in eval_Ns}
            history["evals"].append({"epoch": epoch, "val": val})
            print(f"epoch {epoch}: VAL  " + "  ".join(f"N={n}:{val[n]:.4f}" for n in eval_Ns))
            for n in eval_Ns:
                if val[n] > best[n]:
                    best[n], best_epoch[n] = val[n], epoch
                    save(os.path.join(save_dir, f"best_N_{n}.pt"), epoch, val[n])
                    if n == primary:
                        save(os.path.join(save_dir, "best.pt"), epoch, val[n])
            history["best_per_N"], history["best_per_N_epoch"] = best, best_epoch
            with open(os.path.join(save_dir, "history.json"), "w") as f:
                json.dump(history, f, indent=1)

    # final N-sweep of the selected checkpoint: the only evaluation on the test set
    ckpt = torch.load(os.path.join(save_dir, "best.pt"), map_location=device)
    ema.load_state_dict(ckpt["ema_model"])
    sweep = list(cfg["eval_steps_sweep"])
    val = {n: evaluate(ema, flow, loaders["valid"], device, n, tok.pad_id) for n in sweep}
    test = {n: evaluate(ema, flow, loaders["test"], device, n, tok.pad_id) for n in sweep}
    print(f"\nbest.pt (epoch {ckpt['epoch']}, val@N={primary} {ckpt['val_acc']:.4f}) N-sweep:")
    for n in sweep:
        print(f"  N={n:3d}   val {val[n]:.4f}   test {test[n]:.4f}")
    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump({"best_epoch": ckpt["epoch"], "best_val_acc": ckpt["val_acc"],
                   "val_sweep": val, "test_sweep": test,
                   "best_per_N": best, "best_per_N_epoch": best_epoch}, f, indent=1)


if __name__ == "__main__":
    main()
