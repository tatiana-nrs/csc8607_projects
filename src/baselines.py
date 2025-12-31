# src/baselines.py
import argparse
import random
from collections import Counter

import numpy as np
import torch
import yaml

from src.data_loading import get_dataloaders


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_labels(dataset):
    # HF dataset wrapper fréquent
    if hasattr(dataset, "base_ds"):  # ex: dataset.base_ds["label"]
        return [int(x) for x in dataset.base_ds["label"]]
    if hasattr(dataset, "ds"):
        return [int(x) for x in dataset.ds["label"]]

    # fallback : itération (plus lent)
    labels = []
    for i in range(len(dataset)):
        _, y = dataset[i]
        labels.append(int(y))
    return labels


def _accuracy_from_constant(labels, pred_class: int) -> float:
    labels = np.asarray(labels, dtype=np.int64)
    return float((labels == pred_class).mean())


def _accuracy_random_uniform(labels, num_classes: int, seed: int) -> float:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=np.int64)
    preds = rng.integers(low=0, high=num_classes, size=len(labels), dtype=np.int64)
    return float((preds == labels).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    train_loader, val_loader, test_loader, meta = get_dataloaders(cfg)

    seed = int(cfg["train"]["seed"])
    _set_seed(seed)

    loaders = {"train": train_loader, "val": val_loader, "test": test_loader}
    ds = loaders[args.split].dataset

    num_classes = int(meta["num_classes"])
    labels = _extract_labels(ds)

    counts = Counter(labels)
    majority_class, majority_count = counts.most_common(1)[0]

    acc_majority = _accuracy_from_constant(labels, majority_class)
    acc_random = _accuracy_random_uniform(labels, num_classes=num_classes, seed=seed)

    print(f"[baselines] split={args.split} n={len(labels)} num_classes={num_classes}")
    print(f"[baselines] majority_class={majority_class} count={majority_count}")
    print(f"[baselines] accuracy_majority = {acc_majority:.6f}  ({acc_majority*100:.3f}%)")
    print(f"[baselines] accuracy_random_uniform = {acc_random:.6f}  ({acc_random*100:.3f}%)")
    print(f"[baselines] expected_random_uniform = {1.0/num_classes:.6f}  ({100.0/num_classes:.3f}%)")


if __name__ == "__main__":
    main()
