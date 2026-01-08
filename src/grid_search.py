from __future__ import annotations

"""
Mini grid search (rapide) autour des paramètres trouvés.

Exécution :
    python -m src.grid_search --config configs/config.yaml

Exigences :
- lire la section 'hparams' de la config
- lancer plusieurs runs en variant les hyperparamètres
- journaliser hparams et résultats de chaque run (TensorBoard HParams)
"""

import argparse
import itertools
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.train import set_seed, get_device, build_optimizer, eval_epoch


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def _grid_from_hparams(h: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Construit la liste des configs à tester à partir de hparams.
    Attendu : lr, weight_decay, et 2 hyperparams modèle (ex B, width).
    """
    lr_list = [float(v) for v in _as_list(h.get("lr", []))]
    wd_list = [float(v) for v in _as_list(h.get("weight_decay", []))]

    B_list = _as_list(h.get("B", []))    
    width_list = [float(v) for v in _as_list(h.get("width", []))]

    if not lr_list:
        lr_list = [1e-3]
    if not wd_list:
        wd_list = [1e-4]
    if not B_list:
        B_list = [[2, 2, 2]]
    if not width_list:
        width_list = [1.0]

    combos = []
    for lr, wd, B, width in itertools.product(lr_list, wd_list, B_list, width_list):
        combos.append({"lr": lr, "weight_decay": wd, "B": B, "width": width})
    return combos


def _run_one(
    base_cfg: Dict[str, Any],
    combo: Dict[str, Any],
    device: str,
    runs_dir: Path,
) -> Tuple[float, float]:
    """
    Lance un run court et retourne (best_val_acc, best_val_loss)
    """
    cfg = yaml.safe_load(yaml.safe_dump(base_cfg))

    cfg.setdefault("train", {}).setdefault("optimizer", {})
    cfg["train"]["optimizer"]["lr"] = float(combo["lr"])
    cfg["train"]["optimizer"]["weight_decay"] = float(combo["weight_decay"])

    cfg.setdefault("model", {})
    cfg["model"]["B"] = combo["B"]
    cfg["model"]["width"] = float(combo["width"])

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    #data + model
    train_loader, val_loader, _, meta = get_dataloaders(cfg)
    model = build_model(cfg).to(device)

    optimizer = build_optimizer(model.parameters(), cfg["train"]["optimizer"])
    loss_fn = nn.CrossEntropyLoss()

    epochs = int(cfg.get("train", {}).get("epochs", 3))
    max_steps = cfg.get("train", {}).get("max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None

    # Run name 
    run_name = (
        f"proj_lr={combo['lr']:.0e}_wd={combo['weight_decay']:.0e}"
        f"_B={','.join(map(str, combo['B']))}_width={combo['width']}"
        f"_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    log_dir = runs_dir / run_name
    writer = SummaryWriter(log_dir=str(log_dir))

    hparams_tb = {
        "lr": float(combo["lr"]),
        "weight_decay": float(combo["weight_decay"]),
        "B": str(combo["B"]),
        "width": float(combo["width"]),
        "seed": seed,
        "epochs": epochs,
        "batch_size": int(cfg["train"]["batch_size"]),
    }

    best_val_acc = -1.0
    best_val_loss = float("inf")

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, n = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            b = yb.size(0)
            running_loss += float(loss.item()) * b
            running_correct += float((logits.argmax(1) == yb).float().sum().item())
            n += b

            writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = running_loss / max(1, n)
        train_acc = running_correct / max(1, n)

        num_classes = int(meta.get("num_classes", cfg["model"]["num_classes"]))

        val_loss, val_acc, val_f1 = eval_epoch(
            model,
            val_loader,
            device,
            num_classes=int(meta["num_classes"])
        )

        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("hp/lr", float(combo["lr"]), epoch)
        writer.add_scalar("val/f1", val_f1, epoch)
        writer.add_scalar("hp/weight_decay", float(combo["weight_decay"]), epoch)

        if val_acc > best_val_acc:
            best_val_acc = float(val_acc)
            best_val_loss = float(val_loss)

        if max_steps is not None and global_step >= max_steps:
            break

    metrics_tb = {
        "best_val_acc": best_val_acc,
        "best_val_loss": best_val_loss,
    }

    writer.close()
    print(f"[RUN DONE] {run_name} -> best_val_acc={best_val_acc:.4f} best_val_loss={best_val_loss:.4f}")
    return best_val_acc, best_val_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg: Dict[str, Any] = yaml.safe_load(open(args.config, "r"))

    seed = int(cfg.get("train", {}).get("seed", 42))
    set_seed(seed)

    device = get_device(cfg.get("train", {}))
    print(f"[INFO] device={device} seed={seed}")

    runs_dir = Path(cfg["paths"]["runs_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)

    h = cfg.get("hparams", {}) or {}
    combos = _grid_from_hparams(h)
    print(f"[INFO] grid size = {len(combos)} runs")

    best = None  # (acc, loss, combo)
    for i, combo in enumerate(combos):
        print(f"[GRID {i+1:02d}/{len(combos)}] {combo}")
        acc, loss = _run_one(cfg, combo, device=device, runs_dir=runs_dir)
        if best is None or acc > best[0]:
            best = (acc, loss, combo)

    if best is not None:
        print(f"[BEST OVERALL] acc={best[0]:.4f} loss={best[1]:.4f} combo={best[2]}")
    print("[TIP] TensorBoard -> HParams + Scalars dans runs/")


if __name__ == "__main__":
    main()
