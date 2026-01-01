# src/lr_finder.py
from __future__ import annotations

"""
Recherche de taux d'apprentissage (LR finder).

Exécutable via :
    python -m src.lr_finder --config configs/config.yaml --lr_wd_finder

Exigences :
- lire la config YAML
- produire un log/trace permettant de visualiser (lr, loss) dans TensorBoard
"""

import argparse
import copy
import os
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model


def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(cfg_train: Dict[str, Any]) -> str:
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
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay, nesterov=True)
    return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)


@torch.no_grad()
def evaluate_loss(model: nn.Module, loader: DataLoader, device: str, max_batches: int | None = None) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_n = 0

    for i, (xb, yb) in enumerate(loader):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        b = yb.size(0)
        total_loss += float(loss.item()) * b
        total_n += b
        if max_batches is not None and (i + 1) >= max_batches:
            break

    return total_loss / max(1, total_n)


def make_subset_loader(base_loader: DataLoader, subset_size: int, seed: int, device: str) -> DataLoader:
    ds = base_loader.dataset
    n_total = len(ds)
    n_take = min(max(1, subset_size), n_total)

    g = torch.Generator().manual_seed(seed)
    idxs = torch.randperm(n_total, generator=g).tolist()[:n_take]
    subset = Subset(ds, idxs)

    return DataLoader(
        subset,
        batch_size=base_loader.batch_size,
        shuffle=True,
        num_workers=base_loader.num_workers,
        pin_memory=(device == "cuda"),
        drop_last=False,
    )


def find_best_lr_wd(results: List[Tuple[float, float, float]]) -> Tuple[float, float, Tuple[float, float]]:
    # results: (lr, wd, loss)
    results_sorted = sorted(results, key=lambda x: x[2])
    best_lr, best_wd, min_loss = results_sorted[0]

    # fenêtre "stable" : loss <= min_loss * 1.05 pour le meilleur wd
    thr = min_loss * 1.05
    stable_lrs = [lr for (lr, wd, loss) in results if wd == best_wd and loss <= thr]
    if stable_lrs:
        stable_window = (min(stable_lrs), max(stable_lrs))
    else:
        stable_window = (best_lr, best_lr)

    return best_lr, best_wd, stable_window


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--lr_wd_finder", action="store_true", help="active la recherche conjointe LR/WD")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)
    device = get_device(cfg.get("train", {}))
    print(f"[INFO] device={device} seed={seed}")

    # Dataloaders (on réutilise ton pipeline)
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)

    # --- Params "finder" (pas obligé de les mettre dans YAML, on a des defaults) ---
    finder_cfg = cfg.get("lr_finder", {}) or {}

    subset_size = int(finder_cfg.get("subset_size", 256))
    iters_per_trial = int(finder_cfg.get("iters_per_trial", 10))
    eval_batches = int(finder_cfg.get("eval_batches", 5))  # moyenne sur quelques batches pour stabiliser
    # listes LR/WD : par défaut comme ton exemple + “classiques”
    lr_list = finder_cfg.get("lr_list", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    wd_list = finder_cfg.get("wd_list", [0.0, 1e-5, 1e-4, 1e-3])

    # Optimizer "name" = celui du train.optimizer si présent
    opt_name = str(cfg.get("train", {}).get("optimizer", {}).get("name", "adam"))

    # Subset loader (stabilise + rapide)
    train_loader_small = make_subset_loader(train_loader, subset_size=subset_size, seed=seed, device=device)

    # TensorBoard
    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"lr_wd_finder_{time.strftime('%Y%m%d_%H%M%S')}"
    log_dir = runs_dir / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    loss_fn = nn.CrossEntropyLoss()

    results: List[Tuple[float, float, float]] = []
    trial_idx = 0

    # Pour comparer proprement : on part toujours du même init
    base_model = build_model(cfg).to(device)
    init_state = copy.deepcopy(base_model.state_dict())

    print(f"[INFO] subset_size={subset_size} iters_per_trial={iters_per_trial} eval_batches={eval_batches}")
    print(f"[INFO] testing {len(lr_list) * len(wd_list)} combos (lr x wd)")
    print(f"[INFO] logs -> {log_dir}")

    for wd in wd_list:
        for lr in lr_list:
            # reset model
            base_model.load_state_dict(init_state)
            base_model.train()

            optimizer = build_optimizer(base_model.parameters(), lr=float(lr), weight_decay=float(wd), name=opt_name)

            # petit "warm train" sur quelques itérations
            it = iter(train_loader_small)
            for _ in range(iters_per_trial):
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(train_loader_small)
                    xb, yb = next(it)

                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = base_model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()

            # eval avg_loss (sur qq batches) pour limiter l’oscillation
            avg_loss = evaluate_loss(base_model, train_loader_small, device=device, max_batches=eval_batches)

            print(f"[trial {trial_idx:03d}] lr={lr:.3e} wd={wd:.3e} avg_loss={avg_loss:.4f}")

            # log TB
            writer.add_scalar("lr_finder/loss", avg_loss, trial_idx)
            writer.add_scalar("lr_finder/lr", float(lr), trial_idx)
            writer.add_scalar("lr_finder/wd", float(wd), trial_idx)

            results.append((float(lr), float(wd), float(avg_loss)))
            trial_idx += 1

    writer.close()

    best_lr, best_wd, stable_window = find_best_lr_wd(results)
    print(f"[DONE] logs -> {log_dir}")
    print(f"[BEST] lr={best_lr:.3e} wd={best_wd:.3e}")
    print(f"[STABLE_WINDOW for best wd] [{stable_window[0]:.3e}, {stable_window[1]:.3e}]")
    print("[TIP] TensorBoard: tags lr_finder/lr, lr_finder/wd, lr_finder/loss")


if __name__ == "__main__":
    main()
