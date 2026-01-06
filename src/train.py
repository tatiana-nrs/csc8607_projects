# src/train.py
"""
Entraînement principal.

Exécution :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences :
- lire la config YAML
- respecter paths.runs_dir et paths.artifacts_dir
- logger train/loss et val/loss (tags EXACTS)
- logger une métrique de classification : val/f1 (obligatoire) (+ val/accuracy utile)
- supporter --overfit_small
- sauvegarder le meilleur checkpoint (selon val/f1) dans artifacts/best.ckpt
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, Any, Tuple

import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model


# ---------------- Utils ----------------

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


def build_optimizer(params, cfg_opt: dict):
    name = str(cfg_opt.get("name", "adam")).lower()
    lr = float(cfg_opt.get("lr", 1e-3))
    wd = float(cfg_opt.get("weight_decay", 0.0))
    mom = float(cfg_opt.get("momentum", 0.9))

    if name == "adamw":
        return optim.AdamW(params, lr=lr, weight_decay=wd)
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=mom, weight_decay=wd, nesterov=True)
    return optim.Adam(params, lr=lr, weight_decay=wd)


def make_overfit_loader(train_loader: DataLoader, cfg: dict) -> DataLoader:
    seed = int(cfg["train"].get("seed", 42))
    overfit_n = int(cfg["train"].get("overfit_n", 64))
    overfit_n = max(1, overfit_n)

    ds = train_loader.dataset
    n_total = len(ds)
    n_take = min(overfit_n, n_total)

    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    idxs = perm[:n_take]

    subset = Subset(ds, idxs)

    return DataLoader(
        subset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
        pin_memory=(get_device(cfg["train"]) == "cuda"),
        drop_last=False,
    )


@torch.no_grad()
def _macro_f1_from_preds(preds: torch.Tensor, targets: torch.Tensor, num_classes: int) -> float:
    preds = preds.view(-1).to(torch.int64)
    targets = targets.view(-1).to(torch.int64)

    tp = torch.zeros(num_classes, dtype=torch.float64, device=preds.device)
    fp = torch.zeros(num_classes, dtype=torch.float64, device=preds.device)
    fn = torch.zeros(num_classes, dtype=torch.float64, device=preds.device)

    for c in range(num_classes):
        p_c = (preds == c)
        t_c = (targets == c)
        tp[c] = (p_c & t_c).sum()
        fp[c] = (p_c & ~t_c).sum()
        fn[c] = (~p_c & t_c).sum()

    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)

    return float(f1.mean().item())


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str, num_classes: int) -> Tuple[float, float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0.0
    n = 0

    all_preds = []
    all_targets = []

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        preds = logits.argmax(dim=1)

        b = yb.size(0)
        total_loss += float(loss.item()) * b
        total_correct += float((preds == yb).float().sum().item())
        n += b

        all_preds.append(preds.detach().cpu())
        all_targets.append(yb.detach().cpu())

    avg_loss = total_loss / max(1, n)
    acc = total_correct / max(1, n)

    preds_cat = torch.cat(all_preds, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    f1 = _macro_f1_from_preds(preds_cat, targets_cat, num_classes=num_classes)

    return avg_loss, acc, f1


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    args = parser.parse_args()

    cfg: Dict[str, Any] = yaml.safe_load(open(args.config, "r"))

    # overrides CLI
    if args.seed is not None:
        cfg.setdefault("train", {})["seed"] = args.seed
    if args.overfit_small:
        cfg.setdefault("train", {})["overfit_small"] = True
    if args.max_epochs is not None:
        cfg.setdefault("train", {})["epochs"] = args.max_epochs
    if args.max_steps is not None:
        cfg.setdefault("train", {})["max_steps"] = args.max_steps
    if args.batch_size is not None:
        cfg.setdefault("train", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg.setdefault("train", {}).setdefault("optimizer", {})["lr"] = args.lr
    if args.weight_decay is not None:
        cfg.setdefault("train", {}).setdefault("optimizer", {})["weight_decay"] = args.weight_decay

    seed = int(cfg["train"].get("seed", 42))
    set_seed(seed)

    device = get_device(cfg["train"])
    print(f"[INFO] device = {device}  seed={seed}")

    # Paths
    runs_dir = Path(cfg["paths"]["runs_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = artifacts_dir / "best.ckpt"

    # Data
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)
    num_classes = int(meta.get("num_classes", cfg["model"]["num_classes"]))

    if bool(cfg["train"].get("overfit_small", False)):
        train_loader = make_overfit_loader(train_loader, cfg)
        print(f"[INFO] overfit_small enabled -> train subset size = {len(train_loader.dataset)}")

    # Model
    model = build_model(cfg).to(device)

    # TensorBoard (nom explicite)
    opt_cfg = cfg["train"]["optimizer"]
    lr = float(opt_cfg.get("lr"))
    wd = float(opt_cfg.get("weight_decay"))

    model_cfg = cfg.get("model", {})
    B = model_cfg.get("B", "NA")
    width = model_cfg.get("width", "NA")

    epochs = int(cfg["train"].get("epochs", 1))
    seed = int(cfg["train"].get("seed", 42))

    B_str = ",".join(map(str, B)) if isinstance(B, list) else str(B)

    run_name = (
        f"dsconvnet_"
        f"lr={lr:g}_"
        f"wd={wd:g}_"
        f"B={B_str}_"
        f"w={width}_"
        f"ep={epochs}_"
        f"seed={seed}"
    )

    writer = SummaryWriter(log_dir=str(runs_dir / run_name))
    writer.add_text("config_path", str(args.config))
    writer.add_text("seed", str(seed))

    # Optim / scheduler
    optimizer = build_optimizer(model.parameters(), cfg["train"]["optimizer"])
    sch_cfg = cfg["train"].get("scheduler", {}) or {}
    scheduler = None
    if str(sch_cfg.get("name", "none")).lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg.get("step_size", 10)),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )

    # ===== Loss initiale (AVANT entraînement) =====
    loss_fn = nn.CrossEntropyLoss()
    model.eval()
    xb0, yb0 = next(iter(train_loader))
    xb0, yb0 = xb0.to(device), yb0.to(device)
    with torch.no_grad():
        logits0 = model(xb0)
        init_loss = loss_fn(logits0, yb0).item()

    theo_loss = float(torch.log(torch.tensor(float(num_classes))).item())  # -log(1/C) = log(C)
    print(f"[INIT] batch shape={tuple(xb0.shape)} num_classes={num_classes} "
          f"init_loss={init_loss:.4f} theo_logC={theo_loss:.4f}")

    # ===== Train loop =====
    best_val_f1 = -1.0
    best_val_loss = float("inf")

    max_steps = cfg["train"].get("max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            b = yb.size(0)
            running_loss += float(loss.item()) * b
            seen += b

            global_step += 1
            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = running_loss / max(1, seen)

        val_loss, val_acc, val_f1 = eval_epoch(model, val_loader, device, num_classes=num_classes)

        # logs EXACTS demandés
        writer.add_scalar("train/loss", float(train_loss), epoch)
        writer.add_scalar("val/loss", float(val_loss), epoch)
        writer.add_scalar("val/accuracy", float(val_acc), epoch)
        writer.add_scalar("val/f1", float(val_f1), epoch)
        writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), epoch)

        print(
            f"[{epoch:03d}] train loss {train_loss:.4f} | "
            f"val loss {val_loss:.4f} acc {val_acc:.4f} f1 {val_f1:.4f}"
        )

        improved = (val_f1 > best_val_f1) or (val_f1 == best_val_f1 and val_loss < best_val_loss)
        if improved:
            best_val_f1 = float(val_f1)
            best_val_loss = float(val_loss)
            torch.save({"model": model.state_dict(), "meta": meta, "config": cfg}, best_ckpt)

        if scheduler is not None:
            scheduler.step()

        if max_steps is not None and global_step >= max_steps:
            print(f"[INFO] max_steps atteint ({global_step}), stop.")
            break

    writer.close()
    print(f"[BEST] val f1 = {best_val_f1:.4f} (val loss {best_val_loss:.4f}) -> saved {best_ckpt}")


if __name__ == "__main__":
    main()
