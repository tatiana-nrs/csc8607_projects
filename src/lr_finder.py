from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import yaml

from src.data_loading import get_dataloaders
from src.model import build_model


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg_train: dict) -> str:
    dev = str(cfg_train.get("device", "auto")).lower()
    if dev == "cuda" and torch.cuda.is_available():
        return "cuda"
    if dev == "mps" and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def fixed_subset_loader(loader: DataLoader, subset_size: int, seed: int, shuffle: bool) -> DataLoader:
    ds = loader.dataset
    n = min(len(ds), subset_size)
    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(len(ds), generator=g)[:n].tolist()
    sub = Subset(ds, idxs)
    return DataLoader(
        sub,
        batch_size=loader.batch_size,
        shuffle=shuffle,
        num_workers=0,          # LR finder: simple + stable
        drop_last=False,
        pin_memory=torch.cuda.is_available(),
    )


@torch.no_grad()
def eval_loss(model: nn.Module, loader: DataLoader, device: str, max_batches: int | None = None) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        loss = loss_fn(model(x), y)
        b = y.size(0)
        total += loss.item() * b
        n += b
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return total / max(1, n)


def train_steps(model: nn.Module, loader: DataLoader, device: str, lr: float, wd: float, iters: int) -> None:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    it = iter(loader)
    for _ in range(iters):
        try:
            x, y = next(it)
        except StopIteration:
            it = iter(loader)
            x, y = next(it)

        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--lr_wd_finder", action="store_true")
    p.add_argument("--subset_size", type=int, default=256)
    p.add_argument("--iters_per_trial", type=int, default=100)
    args = p.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)
    device = get_device(cfg["train"])
    print(f"[INFO] device={device} seed={seed}")

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"lr_wd_finder_{time.strftime('%Y%m%d_%H%M%S')}" if args.lr_wd_finder else f"lr_finder_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))

    # IMPORTANT : pour stabilité, évite de mélanger / augmenter pendant finder
    # -> le plus simple : dans ton code data_loading, prévois un flag "disable_augment"
    # Ici, on fait au mieux avec ce que tu as: on prend le loader train existant puis subset fixe.
    train_loader, _, _, meta = get_dataloaders(cfg)

    # subset fixe pour train (shuffle=True ok) + subset fixe pour eval (shuffle=False)
    train_sub = fixed_subset_loader(train_loader, args.subset_size, seed, shuffle=True)
    eval_sub  = fixed_subset_loader(train_loader, args.subset_size, seed, shuffle=False)

    # listes “comme ton camarade”
    lr_list = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    wd_list = [0.0, 1e-5, 1e-4, 1e-3] if args.lr_wd_finder else [float(cfg["train"]["optimizer"].get("weight_decay", 0.0))]

    best = (None, None, float("inf"))

    trial = 0
    for wd in wd_list:
        for lr in lr_list:
            model = build_model(cfg).to(device)

            train_steps(model, train_sub, device, lr=lr, wd=wd, iters=args.iters_per_trial)
            avg = eval_loss(model, eval_sub, device)

            print(f"[trial {trial:03d}] lr={lr:.3e} wd={wd:.3e} avg_loss={avg:.4f}")
            writer.add_scalar("lr_finder/loss", avg, trial)
            writer.add_scalar("lr_finder/lr", lr, trial)
            writer.add_scalar("lr_finder/wd", wd, trial)

            if avg < best[2]:
                best = (lr, wd, avg)
            trial += 1

    writer.close()
    print(f"[DONE] logs -> {runs_dir/run_name}")
    print(f"[BEST] lr={best[0]:.3e} wd={best[1]:.3e} avg_loss={best[2]:.4f}")
    print("[TIP] TensorBoard tags: lr_finder/lr, lr_finder/wd, lr_finder/loss")


if __name__ == "__main__":
    main()
