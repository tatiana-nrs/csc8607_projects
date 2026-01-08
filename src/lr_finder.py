from __future__ import annotations

"""
Recherche de taux d'apprentissage (LR finder) — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.lr_finder --config configs/config.yaml

Exigences minimales :
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard ou équivalent.
"""

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.train import set_seed, get_device, build_optimizer


def make_fixed_subset_loader(
    train_loader: DataLoader,
    subset_size: int,
    seed: int,
    device: str,
) -> DataLoader:
    ds = train_loader.dataset
    n_total = len(ds)
    n_take = min(int(subset_size), n_total)

    g = torch.Generator()
    g.manual_seed(seed)
    idxs = torch.randperm(n_total, generator=g).tolist()[:n_take]
    subset = Subset(ds, idxs)

    batch_size = int(getattr(train_loader, "batch_size", 32) or 32)

    #shuffle=True mais reproductible
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        num_workers=int(getattr(train_loader, "num_workers", 0) or 0),
        pin_memory=(device == "cuda"),
        drop_last=True,
    )


def lr_at(step: int, num_iters: int, min_lr: float, max_lr: float) -> float:
    t = step / max(1, (num_iters - 1))
    return min_lr * (max_lr / min_lr) ** t


def set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def compute_window(lrs: List[float], losses: List[float], frac: float = 1.05) -> Tuple[float, float]:
    """Fenêtre stable: LR dont loss <= min_loss * frac, et on s'arrête avant le min (zone "descente")."""
    if not lrs:
        return (float("nan"), float("nan"))
    min_idx = min(range(len(losses)), key=lambda i: losses[i])
    min_loss = losses[min_idx]
    thr = min_loss * frac

    ok = [i for i in range(0, min_idx + 1) if losses[i] <= thr]
    if not ok:
        return (lrs[min_idx], lrs[min_idx])
    return (min(lrs[i] for i in ok), max(lrs[i] for i in ok))


def lr_range_test(
    cfg: Dict[str, Any],
    base_train_loader: DataLoader,
    device: str,
    seed: int,
    weight_decay: float,
    subset_size: int,
    num_iters: int,
    min_lr: float,
    max_lr: float,
    runs_dir: Path,
) -> None:

    run_name = f"lr_finder_wd{weight_decay:g}_" + time.strftime("%Y%m%d_%H%M%S")
    log_dir = runs_dir / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    print(f"[RUN] wd={weight_decay:g} logs -> {log_dir}")

    train_loader = make_fixed_subset_loader(base_train_loader, subset_size=subset_size, seed=seed, device=device)

    model = build_model(cfg).to(device)
    init_state = copy.deepcopy(model.state_dict())

    opt_cfg = copy.deepcopy(cfg.get("train", {}).get("optimizer", {}) or {})
    opt_cfg["lr"] = float(min_lr)
    opt_cfg["weight_decay"] = float(weight_decay)
    optimizer = build_optimizer(model.parameters(), opt_cfg)

    loss_fn = nn.CrossEntropyLoss()

    model.load_state_dict(init_state)
    model.train()

    lrs: List[float] = []
    raw_losses: List[float] = []
    smooth_losses: List[float] = []

    beta = 0.98
    ema = None
    best = float("inf")

    it = iter(train_loader)

    for step in range(num_iters):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(train_loader)
            xb, yb = next(it)

        lr = lr_at(step, num_iters, min_lr, max_lr)
        set_lr(optimizer, lr)

        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()

        l = float(loss.item())

        if ema is None:
            ema = l
        else:
            ema = beta * ema + (1 - beta) * l
        ema_corr = ema / (1 - beta ** (step + 1))

        lrs.append(lr)
        raw_losses.append(l)
        smooth_losses.append(ema_corr)

        writer.add_scalar("lr_finder/lr", lr, step)
        writer.add_scalar("lr_finder/loss_raw", l, step)
        writer.add_scalar("lr_finder/loss", ema_corr, step)
        writer.add_scalar("lr_finder/wd", float(weight_decay), step)

        best = min(best, ema_corr)

        #arrêt si divergence 
        if step > 10 and ema_corr > 4.0 * best:
            print(f"[STOP] divergence detected at step={step} lr={lr:.3e} loss={ema_corr:.3f}")
            break

    writer.close()

    # proposition LR
    min_idx = min(range(len(smooth_losses)), key=lambda i: smooth_losses[i])
    best_lr = lrs[min_idx]
    suggested_lr = best_lr / 3.0

    wmin, wmax = compute_window(lrs, smooth_losses, frac=1.05)

    print(f"[DONE] wd={weight_decay:g} steps={len(lrs)}")
    print(f"[MIN]  lr_at_min={best_lr:.3e}  loss_min={smooth_losses[min_idx]:.4f}")
    print(f"[SUG]  suggested_lr≈{suggested_lr:.3e} (lr_at_min/3)")
    print(f"[WIN]  stable_window≈[{wmin:.3e}, {wmax:.3e}] (loss <= 1.05 * min, avant le min)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg: Dict[str, Any] = yaml.safe_load(open(args.config, "r"))

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("train", {}))
    print(f"[INFO] device={device} seed={seed}")

    train_loader, _, _, _ = get_dataloaders(cfg)

    train_cfg = cfg.get("train", {}) or {}
    subset_size = int(train_cfg.get("lr_finder_subset", 256))
    num_iters = int(train_cfg.get("lr_finder_iters", 100))

    min_lr = float(train_cfg.get("finder_start_lr", 1e-6))
    max_lr = float(train_cfg.get("finder_end_lr", 1.0)) 

    wd_list = train_cfg.get("lr_finder_weight_decays", None)
    if wd_list is None:
        wd_list = [1e-5, 1e-4]  
    wd_list = [float(x) for x in wd_list]

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] subset_size={subset_size} num_iters={num_iters} min_lr={min_lr:g} max_lr={max_lr:g}")
    print(f"[INFO] weight_decays={wd_list}")
    print("[TIP] TensorBoard tags: lr_finder/lr, lr_finder/loss (et lr_finder/loss_raw)")

    for wd in wd_list:
        lr_range_test(
            cfg=cfg,
            base_train_loader=train_loader,
            device=device,
            seed=seed,
            weight_decay=wd,
            subset_size=subset_size,
            num_iters=num_iters,
            min_lr=min_lr,
            max_lr=max_lr,
            runs_dir=runs_dir,
        )


if __name__ == "__main__":
    main()
