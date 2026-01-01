"""
Recherche de taux d'apprentissage (LR finder).

Exécution :
    python -m src.lr_finder --config configs/config.yaml

Logs TensorBoard :
  - lr_finder/lr   (x-axis = step)
  - lr_finder/loss (x-axis = step)
"""
# src/lr_finder.py
"""
LR finder / LR×WD finder

Exécution :
  1) Sweep LR (WD fixé)
     python -m src.lr_finder --config configs/config.yaml --min_lr 1e-6 --max_lr 1e-1 --num_iters 200 --weight_decay 1e-5

  2) Grid LR×WD (prend les listes dans cfg["hparams"]["lr"] et cfg["hparams"]["weight_decay"])
     python -m src.lr_finder --config configs/config.yaml --lr_wd_finder --subset_size 256 --iters_per_trial 10
"""

from __future__ import annotations

import argparse
import copy
import math
import time
from pathlib import Path
from typing import List, Tuple

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model


def get_device(cfg_train: dict) -> str:
    dev = str(cfg_train.get("device", "auto")).lower()
    if dev in ("cpu", "cuda", "mps"):
        if dev == "cuda" and not torch.cuda.is_available():
            return "cpu"
        if dev == "mps" and not torch.backends.mps.is_available():
            return "cpu"
        return dev
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_subset_loader(train_loader: DataLoader, subset_size: int, seed: int, batch_size: int, device: str, num_workers: int) -> DataLoader:
    ds = train_loader.dataset
    n_total = len(ds)
    n_take = min(max(1, subset_size), n_total)

    g = torch.Generator()
    g.manual_seed(seed)
    idxs = torch.randperm(n_total, generator=g).tolist()[:n_take]
    subset = Subset(ds, idxs)

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device == "cuda"),
        drop_last=True,  # stable steps
    )


@torch.no_grad()
def eval_avg_loss(model: nn.Module, loader: DataLoader, device: str, max_batches: int | None = None) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        b = yb.size(0)
        total += float(loss.item()) * b
        n += b
        if max_batches is not None and (i + 1) >= max_batches:
            break
    return total / max(1, n)


def build_optimizer(model: nn.Module, lr: float, weight_decay: float, name: str = "adam") -> optim.Optimizer:
    name = name.lower()
    if name == "adamw":
        return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


def lr_sweep(cfg: dict, min_lr: float, max_lr: float, num_iters: int, weight_decay: float):
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    device = get_device(cfg["train"])
    print(f"[INFO] device={device} seed={seed}")

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"lr_finder_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))

    train_loader, _, _, _ = get_dataloaders(cfg)

    # petit subset (sinon trop long)
    subset_loader = make_subset_loader(
        train_loader,
        subset_size=int(cfg["train"].get("lr_finder_subset", 256)),
        seed=seed,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        device=device,
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
    )

    model = build_model(cfg).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # LR log-spaced
    lrs = [min_lr * ((max_lr / min_lr) ** (i / max(1, num_iters - 1))) for i in range(num_iters)]

    opt_name = str(cfg["train"]["optimizer"].get("name", "adam"))
    optimizer = build_optimizer(model, lr=lrs[0], weight_decay=weight_decay, name=opt_name)

    it = 0
    for xb, yb in subset_loader:
        if it >= num_iters:
            break
        lr = lrs[it]
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        model.train()
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        writer.add_scalar("lr_finder/lr", lr, it)
        writer.add_scalar("lr_finder/loss", float(loss.item()), it)

        if it % 10 == 0:
            print(f"[{it:04d}] lr={lr:.3e} loss={loss.item():.4f}")
        it += 1

    writer.close()
    print(f"[DONE] logs -> {runs_dir/run_name}")
    print("[TIP] TensorBoard: tags lr_finder/lr et lr_finder/loss")


def lr_wd_grid(cfg: dict, subset_size: int, iters_per_trial: int):
    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    device = get_device(cfg["train"])
    print(f"[INFO] device={device} seed={seed}")

    # on lit les LISTES dans hparams (comme veut ton prof)
    lr_list = cfg.get("hparams", {}).get("lr", None)
    wd_list = cfg.get("hparams", {}).get("weight_decay", None)
    if not isinstance(lr_list, list) or not isinstance(wd_list, list):
        raise ValueError("Pour --lr_wd_finder, il faut hparams.lr = [..] et hparams.weight_decay = [..] dans configs/config.yaml")

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"lr_wd_finder_{time.strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))

    train_loader, _, _, _ = get_dataloaders(cfg)

    subset_loader = make_subset_loader(
        train_loader,
        subset_size=subset_size,
        seed=seed,
        batch_size=int(cfg["train"].get("batch_size", 64)),
        device=device,
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
    )

    # modèle de référence + snapshot pour reset identique
    base_model = build_model(cfg).to(device)
    base_state = copy.deepcopy(base_model.state_dict())
    loss_fn = nn.CrossEntropyLoss()
    opt_name = str(cfg["train"]["optimizer"].get("name", "adam"))

    best = (None, None, float("inf"))  # (lr, wd, loss)

    trial = 0
    for wd in wd_list:
        for lr in lr_list:
            # reset modèle
            base_model.load_state_dict(base_state)

            optimizer = build_optimizer(base_model, lr=float(lr), weight_decay=float(wd), name=opt_name)

            # quelques itérations d’entraînement
            base_model.train()
            it = 0
            for xb, yb in subset_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = base_model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

                it += 1
                if it >= iters_per_trial:
                    break

            # loss moyenne (sur quelques batches du subset)
            avg_loss = eval_avg_loss(base_model, subset_loader, device=device, max_batches=5)

            # logs (même tag, index=trial)
            writer.add_scalar("lr_finder/loss", avg_loss, trial)
            writer.add_scalar("lr_finder/lr", float(lr), trial)
            writer.add_scalar("lr_finder/wd", float(wd), trial)

            print(f"[trial {trial:03d}] lr={float(lr):.3e} wd={float(wd):.3e} avg_loss={avg_loss:.4f}")

            if avg_loss < best[2]:
                best = (float(lr), float(wd), avg_loss)

            trial += 1

    writer.close()
    print(f"[DONE] logs -> {runs_dir/run_name}")
    print(f"[BEST] lr={best[0]:.3e} wd={best[1]:.3e} avg_loss={best[2]:.4f}")
    print("[TIP] TensorBoard: tags lr_finder/lr, lr_finder/wd, lr_finder/loss")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    # mode sweep
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=1e-1)
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=1e-5)

    # mode grid
    parser.add_argument("--lr_wd_finder", action="store_true")
    parser.add_argument("--subset_size", type=int, default=256)
    parser.add_argument("--iters_per_trial", type=int, default=10)

    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))

    if args.lr_wd_finder:
        lr_wd_grid(cfg, subset_size=args.subset_size, iters_per_trial=args.iters_per_trial)
    else:
        lr_sweep(cfg, min_lr=args.min_lr, max_lr=args.max_lr, num_iters=args.num_iters, weight_decay=args.weight_decay)


if __name__ == "__main__":
    main()
