"""
Recherche de taux d'apprentissage (LR finder).

Exécution :
    python -m src.lr_finder --config configs/config.yaml

Logs TensorBoard :
  - lr_finder/lr   (x-axis = step)
  - lr_finder/loss (x-axis = step)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model


def set_seed(seed: int | None):
    if seed is None:
        return
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def build_optimizer(params, lr: float, weight_decay: float, name: str = "adam"):
    name = str(name).lower()
    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return optim.Adam(params, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def _infinite(loader):
    while True:
        for batch in loader:
            yield batch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)

    # paramètres LR finder (raisonnables par défaut)
    parser.add_argument("--min_lr", type=float, default=1e-6)
    parser.add_argument("--max_lr", type=float, default=1.0)
    parser.add_argument("--num_iters", type=int, default=200)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    # overrides
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    device = get_device(cfg.get("train", {}))
    print(f"[INFO] device={device} seed={seed}")

    # data
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)

    # model
    model = build_model(cfg).to(device)
    model.train()

    # TB
    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime("lr_finder_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))
    writer.add_text("config_path", str(args.config), 0)
    writer.add_text("meta", str(meta), 0)

    # LR schedule log-scale : lr_i = min_lr * (max_lr/min_lr)^(i/(N-1))
    min_lr = float(args.min_lr)
    max_lr = float(args.max_lr)
    N = int(args.num_iters)

    if N < 2:
        raise ValueError("--num_iters doit être >= 2")

    gamma = (max_lr / min_lr) ** (1.0 / (N - 1))  # multiplicateur par itération

    # optimizer (on va changer lr à chaque itération)
    opt_name = cfg.get("train", {}).get("optimizer", {}).get("name", "adam")
    wd = float(args.weight_decay)
    optimizer = build_optimizer(model.parameters(), lr=min_lr, weight_decay=wd, name=opt_name)

    loss_fn = nn.CrossEntropyLoss()

    it = _infinite(train_loader)

    best_loss = float("inf")
    for step in range(N):
        lr = min_lr * (gamma ** step)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        xb, yb = next(it)
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        loss_val = float(loss.item())
        best_loss = min(best_loss, loss_val)

        # logs
        writer.add_scalar("lr_finder/lr", lr, step)
        writer.add_scalar("lr_finder/loss", loss_val, step)

        # arrêt si divergence nette
        if math.isfinite(loss_val) is False or loss_val > 10.0 * best_loss:
            print(f"[STOP] divergence step={step} lr={lr:.3e} loss={loss_val:.4f} (best={best_loss:.4f})")
            break

        if step % 10 == 0:
            print(f"[{step:04d}] lr={lr:.3e} loss={loss_val:.4f}")

    writer.close()
    print(f"[DONE] logs -> {runs_dir / run_name}")
    print("[TIP] TensorBoard: cherche les tags lr_finder/lr et lr_finder/loss")


if __name__ == "__main__":
    main()
