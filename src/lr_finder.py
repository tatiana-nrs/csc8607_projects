# src/lr_finder.py
"""
LR finder (LR + weight decay) compatible avec ton dépôt.

Exécution (comme veut le prof) :
    python -m src.lr_finder --config configs/config.yaml

Principe :
- lit la config YAML
- prend les listes dans config["hparams"]["lr"] et config["hparams"]["weight_decay"]
- entraîne très brièvement sur un sous-ensemble fixe (pour réduire l’oscillation)
- log dans TensorBoard : lr_finder/loss, lr_finder/lr, lr_finder/wd
- affiche BEST + fenêtre "stable" (loss <= min_loss * 1.05) pour le meilleur wd
"""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.train import set_seed, get_device, build_optimizer  # réutilise tes utilitaires


def _fixed_subset_loader(train_loader: DataLoader, subset_size: int, seed: int, device: str) -> DataLoader:
    """Sous-ensemble déterministe pour limiter les variations (oscillations)."""
    ds = train_loader.dataset
    n_total = len(ds)
    n_take = min(int(subset_size), n_total)

    g = torch.Generator()
    g.manual_seed(seed)
    idxs = torch.randperm(n_total, generator=g).tolist()[:n_take]
    subset = Subset(ds, idxs)

    return DataLoader(
        subset,
        batch_size=int(train_loader.batch_size),
        shuffle=True,
        num_workers=int(getattr(train_loader, "num_workers", 0) or 0),
        pin_memory=(device == "cuda"),
        drop_last=False,
    )


@torch.no_grad()
def _avg_loss(model: nn.Module, loader: DataLoader, device: str, max_batches: int) -> float:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total, n = 0.0, 0
    for i, (xb, yb) in enumerate(loader):
        if i >= max_batches:
            break
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        b = yb.size(0)
        total += float(loss.item()) * b
        n += b
    return total / max(1, n)


def _train_few_iters(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    iters: int,
) -> None:
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    it = iter(loader)
    for _ in range(iters):
        try:
            xb, yb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb = next(it)

        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        loss.backward()
        optimizer.step()


def _stable_window_for_wd(results: List[Tuple[float, float, float]], best_wd: float) -> Tuple[float, float]:
    """Fenêtre stable = LR dont la loss est <= min_loss_wd * 1.05 (pour ce wd)."""
    wd_rows = [(lr, loss) for (lr, wd, loss) in results if wd == best_wd]
    wd_rows.sort(key=lambda x: x[0])
    min_loss = min(loss for _, loss in wd_rows)
    thr = min_loss * 1.05
    stable_lrs = [lr for lr, loss in wd_rows if loss <= thr]
    if not stable_lrs:
        # fallback
        best_lr = min(wd_rows, key=lambda x: x[1])[0]
        return best_lr, best_lr
    return min(stable_lrs), max(stable_lrs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg: Dict[str, Any] = yaml.safe_load(open(args.config, "r"))

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    device = get_device(cfg.get("train", {}))
    print(f"[INFO] device={device} seed={seed}")

    # Data loaders (comme train.py)
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)

    # Paramètres LR finder (pas besoin d’ajouter dans YAML)
    subset_size = int(cfg.get("train", {}).get("lr_finder_subset", 256))
    iters_per_trial = int(cfg.get("train", {}).get("lr_finder_iters", 10))
    eval_batches = int(cfg.get("train", {}).get("lr_finder_eval_batches", 5))

    print(f"[INFO] subset_size={subset_size} iters_per_trial={iters_per_trial} eval_batches={eval_batches}")

    # Grilles depuis hparams (comme ton camarade)
    h = cfg.get("hparams", {}) or {}
    lr_list = h.get("lr", [1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    wd_list = h.get("weight_decay", [0.0, 1e-5, 1e-4, 1e-3])

    # Normalise en float
    lr_list = [float(x) for x in lr_list]
    wd_list = [float(x) for x in wd_list]

    # Sous-ensemble fixe pour stabilité
    finder_loader = _fixed_subset_loader(train_loader, subset_size=subset_size, seed=seed, device=device)

    # Logs
    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_name = time.strftime("lr_wd_finder_%Y%m%d_%H%M%S")
    log_dir = runs_dir / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[INFO] testing {len(lr_list) * len(wd_list)} combos (lr x wd)")
    print(f"[INFO] logs -> {log_dir}")

    # Base model state (réinit à chaque combo)
    base_model = build_model(cfg).to(device)
    base_state = copy.deepcopy(base_model.state_dict())

    results: List[Tuple[float, float, float]] = []
    trial = 0

    for wd in wd_list:
        for lr in lr_list:
            # reset weights
            base_model.load_state_dict(base_state)

            # build optimizer en forçant lr/wd (sans toucher au YAML)
            opt_cfg = copy.deepcopy(cfg.get("train", {}).get("optimizer", {}) or {})
            opt_cfg["lr"] = lr
            opt_cfg["weight_decay"] = wd
            optimizer = build_optimizer(base_model.parameters(), opt_cfg)

            # tiny train
            _train_few_iters(base_model, finder_loader, optimizer, device=device, iters=iters_per_trial)

            # eval (moyenne sur quelques batches)
            avg_loss = _avg_loss(base_model, finder_loader, device=device, max_batches=eval_batches)

            print(f"[trial {trial:03d}] lr={lr:.3e} wd={wd:.3e} avg_loss={avg_loss:.4f}")
            writer.add_scalar("lr_finder/loss", avg_loss, trial)
            writer.add_scalar("lr_finder/lr", lr, trial)
            writer.add_scalar("lr_finder/wd", wd, trial)

            results.append((lr, wd, avg_loss))
            trial += 1

    writer.close()

    # BEST
    best_lr, best_wd, best_loss = min(results, key=lambda t: t[2])
    wmin, wmax = _stable_window_for_wd(results, best_wd)

    print(f"[DONE] logs -> {log_dir}")
    print(f"[BEST] lr={best_lr:.3e} wd={best_wd:.3e} avg_loss={best_loss:.4f}")
    print(f"[STABLE_WINDOW for best wd] [{wmin:.3e}, {wmax:.3e}]")
    print("[TIP] TensorBoard: tags lr_finder/lr, lr_finder/wd, lr_finder/loss")


if __name__ == "__main__":
    main()
