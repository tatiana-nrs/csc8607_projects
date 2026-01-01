# src/train.py
"""
Entraînement principal (à implémenter par l'étudiant·e).

Exécution :
    python -m src.train --config configs/config.yaml [--seed 42]

Exigences :
- lire la config YAML
- respecter paths.runs_dir et paths.artifacts_dir
- logger train/loss et val/loss (+ au moins une métrique classification)
- supporter --overfit_small (sur-apprendre sur un très petit échantillon)
"""

import argparse
import math
import time
from pathlib import Path

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
    # auto
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


def grad_norm_sum(model: nn.Module) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        total += float(torch.norm(g, p=2).item())
    return total


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: str):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, n = 0.0, 0.0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        preds = logits.argmax(dim=1)

        b = yb.size(0)
        total_loss += loss.item() * b
        total_correct += (preds == yb).float().sum().item()
        n += b

    return total_loss / max(1, n), total_correct / max(1, n)


def make_overfit_loader(train_loader: DataLoader, cfg: dict) -> DataLoader:
    """
    Tronque le dataset train pour sur-apprendre rapidement.
    Par défaut: 64 exemples (ou cfg["train"]["overfit_n"] si présent).
    """
    seed = int(cfg["train"].get("seed", 42))
    overfit_n = int(cfg["train"].get("overfit_n", 64))  # <-- tu peux ajouter ça dans le YAML si tu veux
    overfit_n = max(1, overfit_n)

    ds = train_loader.dataset
    n_total = len(ds)
    n_take = min(overfit_n, n_total)

    # indices déterministes (reproductible)
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n_total, generator=g).tolist()
    idxs = perm[:n_take]

    subset = Subset(ds, idxs)

    # IMPORTANT: shuffle=True pour overfit (on s’en fiche, mais ça aide)
    return DataLoader(
        subset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["dataset"].get("num_workers", 0)),
        pin_memory=(get_device(cfg["train"]) == "cuda"),
        drop_last=False,
    )


def first_batch_check(model: nn.Module, train_loader: DataLoader, device: str, num_classes: int):
    """
    - forward sur 1 batch
    - loss initiale
    - backward 1 step (sans optimizer.step) pour vérifier grads non nuls
    """
    model.train()
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    logits = model(xb)
    loss = nn.CrossEntropyLoss()(logits, yb)

    expected = -math.log(1.0 / float(num_classes))  # ~ log(num_classes) si logits ~0
    model.zero_grad(set_to_none=True)
    loss.backward()

    gn = grad_norm_sum(model)
    nonzero = (gn > 0.0)

    info = {
        "x_shape": tuple(xb.shape),
        "y_shape": tuple(yb.shape),
        "logits_shape": tuple(logits.shape),
        "loss": float(loss.item()),
        "expected": float(expected),
        "grad_norm_sum": float(gn),
        "grads_nonzero": bool(nonzero),
    }
    return info


# ---------------- Main ----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--overfit_small", action="store_true")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--max_steps", type=int, default=None)
    # overrides utiles
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    args = parser.parse_args()

    # 1) Load config
    cfg = yaml.safe_load(open(args.config, "r"))

    # 2) Overrides CLI (sans réécrire en dur des valeurs)
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

    # 3) Paths
    runs_dir = Path(cfg["paths"]["runs_dir"])
    artifacts_dir = Path(cfg["paths"]["artifacts_dir"])
    runs_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = artifacts_dir / "best.ckpt"  # nom exigé

    # 4) Data
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)
    num_classes = int(meta.get("num_classes", cfg["model"]["num_classes"]))

    # overfit_small => sous-ensemble minuscule
    if bool(cfg["train"].get("overfit_small", False)):
        train_loader = make_overfit_loader(train_loader, cfg)
        print(f"[INFO] overfit_small enabled -> train subset size = {len(train_loader.dataset)}")

    print("MODEL CFG:", cfg.get("model", {}))
    print("HPARAMS:", cfg.get("hparams", {}))
    
    # 5) Model
    model = build_model(cfg).to(device)

    # 6) TensorBoard
    run_name = time.strftime("run_%Y%m%d_%H%M%S")
    writer = SummaryWriter(log_dir=str(runs_dir / run_name))
    writer.add_text("meta", str(meta))
    writer.add_text("config_path", str(args.config))
    writer.add_text("seed", str(seed))

    # 7) Check premier batch + loss initiale + grads
    try:
        chk = first_batch_check(model, train_loader, device, num_classes=num_classes)
        print(f"[check] batch x shape = {chk['x_shape']}  y shape = {chk['y_shape']}")
        print(f"[check] logits shape  = {chk['logits_shape']}")
        print(f"[check] initial loss  = {chk['loss']:.6f}")
        print(f"[check] expected ~ -log(1/{num_classes}) = {chk['expected']:.6f}")
        print(f"[check] grad_norm_sum = {chk['grad_norm_sum']:.6f}  nonzero={chk['grads_nonzero']}")
        writer.add_scalar("check/initial_loss", chk["loss"], 0)
        writer.add_scalar("check/expected_loss", chk["expected"], 0)
        writer.add_scalar("check/grad_norm_sum", chk["grad_norm_sum"], 0)
    except StopIteration:
        print("[WARN] train_loader vide ?")
        writer.close()
        return

    # 8) Optim / scheduler
    optimizer = build_optimizer(model.parameters(), cfg["train"]["optimizer"])
    sch_cfg = cfg["train"].get("scheduler", {}) or {}
    scheduler = None
    if str(sch_cfg.get("name", "none")).lower() == "step":
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(sch_cfg.get("step_size", 10)),
            gamma=float(sch_cfg.get("gamma", 0.1)),
        )

    # 9) Train loop
    loss_fn = nn.CrossEntropyLoss()
    best_val_acc = -1.0
    epochs = int(cfg["train"].get("epochs", 1))
    max_steps = cfg["train"].get("max_steps", None)
    max_steps = int(max_steps) if max_steps is not None else None

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_correct, seen = 0.0, 0.0, 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)

            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            b = yb.size(0)
            running_loss += loss.item() * b
            running_correct += (logits.argmax(1) == yb).float().sum().item()
            seen += b

            # log step (utile surtout en overfit_small)
            writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            global_step += 1

            if max_steps is not None and global_step >= max_steps:
                break

        train_loss = running_loss / max(1, seen)
        train_acc = running_correct / max(1, seen)

        val_loss, val_acc = eval_epoch(model, val_loader, device)

        # logs epoch
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("lr", float(optimizer.param_groups[0]["lr"]), epoch)

        print(f"[{epoch:03d}] train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f}")

        # save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model": model.state_dict(), "meta": meta, "config": cfg}, best_ckpt)

        if scheduler is not None:
            scheduler.step()

        if max_steps is not None and global_step >= max_steps:
            print(f"[INFO] max_steps atteint ({global_step}), stop.")
            break

    writer.close()
    print(f"[BEST] val acc = {best_val_acc:.4f}  -> saved {best_ckpt}")


if __name__ == "__main__":
    main()
