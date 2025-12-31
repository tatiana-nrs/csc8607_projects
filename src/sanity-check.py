# src/sanity_check.py
import argparse
import os
from typing import Dict, Any, Tuple, Optional

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torchvision.utils import make_grid, save_image
import torchvision.transforms as T
import yaml
from PIL import Image

from src.data_loading import get_dataloaders
from src.model import build_model


def _to_pil(img):
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)


def _unnormalize(x: torch.Tensor, mean, std) -> torch.Tensor:
    # x: (3,H,W) normalized
    mean_t = torch.tensor(mean, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=x.dtype, device=x.device).view(-1, 1, 1)
    y = x * std_t + mean_t
    return y.clamp(0.0, 1.0)


def _get_base_hf_dataset(ds):
    # try common attribute names
    for name in ["base_ds", "hf_ds", "ds", "dataset"]:
        if hasattr(ds, name):
            return getattr(ds, name)
    return None


def _is_shuffled(loader) -> Tuple[bool, str]:
    s = getattr(loader, "sampler", None)
    if s is None:
        return False, "no_sampler"
    if isinstance(s, RandomSampler):
        return True, "RandomSampler"
    if isinstance(s, SequentialSampler):
        return False, "SequentialSampler"
    # DistributedSampler or others
    if hasattr(s, "shuffle"):
        return bool(getattr(s, "shuffle")), f"{type(s).__name__}(shuffle={getattr(s, 'shuffle')})"
    return False, type(s).__name__


def _save_before_after(
    split_name: str,
    loader,
    out_path: str,
    n: int,
    mean,
    std,
    train_has_aug: bool,
):
    ds = loader.dataset
    base = _get_base_hf_dataset(ds)

    to_tensor = T.ToTensor()
    raw_imgs = []
    proc_imgs = []

    n = min(n, len(ds))
    for i in range(n):
        # RAW
        if base is not None:
            raw = base[i]["image"]
        else:
            # fallback: if we can't access raw, at least show processed only
            raw = None

        if raw is not None:
            raw = _to_pil(raw).convert("RGB")
            raw_t = to_tensor(raw)  # (3,H,W) in [0,1]
            raw_imgs.append(raw_t)

        # PROCESSED (train: aug+preprocess, val/test: preprocess)
        x, _ = ds[i]  # x normalized tensor (3,H,W)
        if isinstance(x, torch.Tensor) and x.ndim == 3:
            x_vis = _unnormalize(x.cpu(), mean, std)
            proc_imgs.append(x_vis)

    if not proc_imgs:
        return

    # Build grid: top row raw, bottom row processed
    if raw_imgs:
        raw_grid = make_grid(raw_imgs, nrow=n, padding=2)
        proc_grid = make_grid(proc_imgs, nrow=n, padding=2)
        both = torch.cat([raw_grid, proc_grid], dim=1)  # stack vertically (H axis)
    else:
        both = make_grid(proc_imgs, nrow=n, padding=2)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(both, out_path)

    tag = "raw vs aug+preprocess" if (split_name == "train" and train_has_aug) else "raw vs preprocess"
    print(f"[sanity] saved: {out_path}  ({split_name}: {tag})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--n", type=int, default=8)
    args = p.parse_args()

    cfg: Dict[str, Any] = yaml.safe_load(open(args.config, "r"))

    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)
    print("[sanity] meta =", meta)

    # ===== CHECK 1: batch shapes vs meta["input_shape"] =====
    xb, yb = next(iter(train_loader))
    xv, yv = next(iter(val_loader))
    print("[sanity] train batch:", tuple(xb.shape), tuple(yb.shape))
    print("[sanity] val batch  :", tuple(xv.shape), tuple(yv.shape))

    expected = tuple(meta["input_shape"])
    if tuple(xb.shape[1:]) != expected:
        raise ValueError(f"[sanity] train batch shape mismatch: got {tuple(xb.shape[1:])}, expected {expected}")
    if tuple(xv.shape[1:]) != expected:
        raise ValueError(f"[sanity] val batch shape mismatch: got {tuple(xv.shape[1:])}, expected {expected}")

    # ===== CHECK 2: shuffle train only =====
    tr_shuf, tr_kind = _is_shuffled(train_loader)
    va_shuf, va_kind = _is_shuffled(val_loader)
    te_shuf, te_kind = _is_shuffled(test_loader)
    print(f"[sanity] shuffle check: train={tr_shuf} ({tr_kind}), val={va_shuf} ({va_kind}), test={te_shuf} ({te_kind})")
    if not tr_shuf:
        raise ValueError("[sanity] train_loader is not shuffled (expected shuffle=True).")
    if va_shuf or te_shuf:
        raise ValueError("[sanity] val/test loaders are shuffled (expected shuffle=False).")

    # ===== CHECK 3: labels/logits coherence (range + output dim) =====
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.eval()

    xb = xb.to(device)
    with torch.no_grad():
        logits = model(xb)

    num_classes = int(meta["num_classes"])
    if logits.ndim != 2 or logits.shape[0] != xb.shape[0] or logits.shape[1] != num_classes:
        raise ValueError(
            f"[sanity] logits shape mismatch: got {tuple(logits.shape)}, expected (batch, {num_classes})."
        )

    yb_cpu = yb.detach().cpu()
    y_min = int(yb_cpu.min().item())
    y_max = int(yb_cpu.max().item())
    if y_min < 0 or y_max >= num_classes:
        raise ValueError(f"[sanity] label range invalid: min={y_min}, max={y_max}, expected in [0, {num_classes-1}]")
    print(f"[sanity] logits OK: {tuple(logits.shape)} | labels OK: range=[{y_min},{y_max}] within 0..{num_classes-1}")

    # ===== Save before/after images into artifacts =====
    mean = cfg["preprocess"]["normalize"]["mean"]
    std = cfg["preprocess"]["normalize"]["std"]
    out_dir = os.path.join(cfg["paths"]["artifacts_dir"], "sanity")
    train_aug_enabled = bool((cfg.get("augment", {}) or {}).get("random_flip", False) or (cfg.get("augment", {}) or {}).get("random_crop", None))

    _save_before_after(
        "train",
        train_loader,
        os.path.join(out_dir, "before_after_train.png"),
        args.n,
        mean,
        std,
        train_has_aug=train_aug_enabled,
    )
    _save_before_after(
        "val",
        val_loader,
        os.path.join(out_dir, "before_after_val.png"),
        args.n,
        mean,
        std,
        train_has_aug=False,
    )

    print("[sanity] done. Check:", out_dir)


if __name__ == "__main__":
    main()
