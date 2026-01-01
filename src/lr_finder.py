from __future__ import annotations

"""
LR finder (sweep LR log-scale) compatible avec ton dépôt.

Exécution (comme veut le prof) :
    python -m src.lr_finder --config configs/config.yaml

Principe (méthode "LR range test") :
- on prend un sous-ensemble fixe
- on fait varier le LR de min_lr -> max_lr (log scale) sur ~num_iters
- on log à chaque itération : (lr, loss) => TensorBoard tags:
    lr_finder/lr
    lr_finder/loss
- on répète pour plusieurs weight_decay (runs séparés pour comparer facilement)
"""

import argparse
import copy
import math
import time
from pathlib import Path
from typing import Any, Dict, List

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.train import set_seed, get_device, build_optimizer


def _make_fixed_subset_loader(train_loader: DataLoader, subset_size: int, seed: int, device: str) -> DataLoader:
    ds = train_loader.dataset
    n_total = len(ds)
    n_take = min(int(subset_size), n_total)

    g = torch.Generator()
    g.manual_seed(seed)
    idxs = torch.randperm(n_total, generator=g).tolist()[:n_take]
    subset = Subset(ds, idxs)

    # IMPORTANT pour réduire l’oscillation:
    # - shuffle=False => séquence stable (reproductible)
    # - drop_last=True => batch shape stable
    batch_size = int(train_loader.batch_size) if getattr(train_loader, "batch_size", None) else 32
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=int(getattr(train_loader, "num_workers", 0) or 0),
        pin_memory=(device == "cuda"),
        drop_last=True,
    )


def _lr_at(step: int, num_iters: int, min_lr: float, max_lr: float) -> float:
    # log-scale interpolation
    t = step / max(1, (num_iters - 1))
    return min_lr * (max_lr / min_lr) ** t


def _set_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for pg in optimizer.param_groups:
        pg["lr"] = lr


def lr_sweep_one_wd(
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    device: str,
    *,
    weight_decay: float,
    min_lr: float,
    max_lr: float,
    num_iters: int,
    log_dir: Path,
    seed: int,
) -> None:
    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("config_path", str(cfg.get("__config_path__", "")))
    writer.add_text("lr_finder/weight_decay", str(weight_decay))
    writer.add_scalar("lr_finder/wd", float(weight_decay), 0)

    model = build_model(cfg).to(device)
    base_state = copy.deepcopy(model.state_dict())

    # build optimizer (on force wd ici, et on override lr à chaque step)
    opt_cfg = copy.deepcopy(cfg.get("train", {}).get("optimizer", {}) or {})
    opt_cfg["weight_decay"] = float(weight_decay)
    opt_cfg["lr"] = float(min_lr)
    optimizer = build_optimizer(model.parameters(), opt_cfg)

    loss_fn = nn.CrossEntropyLoss()
    model.load_state_dict(base_state)
    model.train()

    best = float("inf")
    ema = None
    beta = 0.98  # smoothing pour la courbe

    it = iter(train_loader)

    for step in range(num_iters):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(train_loader)
            xb, yb = next(it)

        lr = _lr_at(step, num_iters, min_lr, max_lr)
        _set_lr(optimizer, lr)

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

        # bias correction (comme souvent dans LR finder)
        ema_corr = ema / (1 - beta ** (step + 1))

        best = min(best, ema_corr)

        writer.add_scalar("lr_finder/lr", lr, step)
        writer.add_scalar("lr_finder/loss", ema_corr, step)

        # arrêt anticipé si explosion (sinon ça “écrase” le graphe)
        if step > 10 and ema_corr > 4.0 * best:
            break

    writer.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg: Dict[str, Any] = yaml.safe_load(open(args.config, "r"))
    cfg["__config_path__"] = args.config  # juste pour log

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    device = get_device(cfg.get("train", {}))
    print(f"[INFO] device={device} seed={seed}")

    # loaders
    train_loader, _, _, _ = get_dataloaders(cfg)

    # paramètres LR finder (peuvent être dans YAML, sinon defaults "cours")
    train_cfg = cfg.get("train", {}) or {}
    subset_size = int(train_cfg.get("lr_finder_subset", 2048))
    num_iters = int(train_cfg.get("lr_finder_iters", 100))  # demandé ~100
    min_lr = float(train_cfg.get("finder_start_lr", 1e-6))
    max_lr = float(train_cfg.get("finder_end_lr", 1.0))

    # weight decay list : on privilégie 1e-5 / 1e-4 comme attendu
    wd_list = train_cfg.get("lr_finder_weight_decays", None)
    if wd_list is None:
        # fallback : si config.hparams.weight_decay existe, on l'utilise,
        # sinon on prend {1e-5, 1e-4}
        wd_list = (cfg.get("hparams", {}) or {}).get("weight_decay", [1e-5, 1e-4])
    wd_list = [float(x) for x in wd_list]

    # subset fixe (réduit l'oscillation)
    finder_loader = _make_fixed_subset_loader(train_loader, subset_size=subset_size, seed=seed, device=device)

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] subset_size={subset_size} num_iters={num_iters} min_lr={min_lr:g} max_lr={max_lr:g}")
    print(f"[INFO] weight_decays={wd_list}")

    # un run séparé par wd => TB compare facilement
    for wd in wd_list:
        run_name = f"lr_finder_wd{wd:g}_" + time.strftime("%Y%m%d_%H%M%S")
        log_dir = runs_dir / run_name
        print(f"[RUN] wd={wd:g} logs -> {log_dir}")
        lr_sweep_one_wd(
            cfg,
            finder_loader,
            device,
            weight_decay=wd,
            min_lr=min_lr,
            max_lr=max_lr,
            num_iters=num_iters,
            log_dir=log_dir,
            seed=seed,
        )

    print("[DONE] TensorBoard: regarde les tags lr_finder/lr et lr_finder/loss (plusieurs runs, un par wd)")


if __name__ == "__main__":
    main()
