# src/data_loading.py
"""
Chargement des données.

Signature imposée :
get_dataloaders(config: dict) -> (train_loader, val_loader, test_loader, meta: dict)

meta doit contenir au minimum :
- "num_classes": int
- "input_shape": tuple
"""

from typing import Dict, Any, Tuple
import argparse
import yaml

import torch
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset

from src.preprocessing import get_preprocess_transforms
from src.augmentation import get_augmentation_transforms


class HFDatasetTorch(Dataset):
    """Wrapper HF Dataset -> PyTorch Dataset avec transforms."""
    def __init__(
        self,
        hf_ds,
        image_key: str = "image",
        label_key: str = "label",
        preprocess=None,
        augment=None,
        is_train: bool = False,
    ):
        self.hf_ds = hf_ds
        self.image_key = image_key
        self.label_key = label_key
        self.preprocess = preprocess
        self.augment = augment
        self.is_train = is_train

    def __len__(self):
        return len(self.hf_ds)

    def __getitem__(self, idx):
        item = self.hf_ds[int(idx)]
        img = item[self.image_key]
        y = item[self.label_key]

        # Augmentations : train uniquement
        if self.is_train and self.augment is not None:
            img = self.augment(img)

        # Preprocess : train/val/test (invariant)
        if self.preprocess is not None:
            img = self.preprocess(img)

        # label int
        y = int(y)
        return img, y


def get_dataloaders(config: Dict[str, Any]):
    ds_cfg = config.get("dataset", {}) or {}
    train_cfg = config.get("train", {}) or {}
    model_cfg = config.get("model", {}) or {}

    ds_name = ds_cfg.get("name")
    if not ds_name:
        raise ValueError("configs/config.yaml: dataset.name est requis (ex: zh-plus/tiny-imagenet).")

    # Splits HF (chez toi: train + valid ; test = null => on le crée)
    split_cfg = ds_cfg.get("split", {}) or {}
    train_split_name = split_cfg.get("train", "train")
    val_split_name = split_cfg.get("val", "valid")
    test_split_name = split_cfg.get("test", None)  # null => créer split

    seed = int(train_cfg.get("seed", 42))
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(ds_cfg.get("num_workers", 0))
    shuffle = bool(ds_cfg.get("shuffle", True))

    # Colonnes (TinyImageNet HF: image/label) — configurable si besoin
    image_key = ds_cfg.get("image_key", "image")
    label_key = ds_cfg.get("label_key", "label")

    # Cache HF (ton YAML: dataset.root = "./data")
    cache_dir = ds_cfg.get("root", None)

    # 1) Charger splits HF
    hf_train_full = load_dataset(ds_name, split=train_split_name, cache_dir=cache_dir)
    hf_val = load_dataset(ds_name, split=val_split_name, cache_dir=cache_dir)

    # 2) Créer test si absent
    if test_split_name is None:
        # lisible depuis YAML si tu veux : dataset.test_size, sinon défaut 0.1
        test_size = float(ds_cfg.get("test_size", 0.1))

        split = hf_train_full.train_test_split(
            test_size=test_size,
            seed=seed,
            stratify_by_column=label_key,
        )
        hf_train = split["train"]
        hf_test = split["test"]
    else:
        hf_train = hf_train_full
        hf_test = load_dataset(ds_name, split=test_split_name, cache_dir=cache_dir)

    # 3) Transforms
    preprocess_tf = get_preprocess_transforms(config)     # callable
    augment_tf = get_augmentation_transforms(config)      # callable ou None (train-only)

    # 4) Wrappers Torch
    train_ds = HFDatasetTorch(
        hf_train, image_key=image_key, label_key=label_key,
        preprocess=preprocess_tf, augment=augment_tf, is_train=True
    )
    val_ds = HFDatasetTorch(
        hf_val, image_key=image_key, label_key=label_key,
        preprocess=preprocess_tf, augment=None, is_train=False
    )
    test_ds = HFDatasetTorch(
        hf_test, image_key=image_key, label_key=label_key,
        preprocess=preprocess_tf, augment=None, is_train=False
    )

    # 5) DataLoaders
    pin_memory = bool(torch.cuda.is_available())
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    # 6) meta (min requis + infos utiles)
    num_classes = int(model_cfg.get("num_classes", 200))
    input_shape = tuple(model_cfg.get("input_shape", (3, 64, 64)))

    meta = {
        "num_classes": num_classes,
        "input_shape": input_shape,
        "seed": seed,
        "sizes": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
    }

    return train_loader, val_loader, test_loader, meta


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, "r"))
    tr, va, te, meta = get_dataloaders(cfg)

    print("meta =", meta)
    xb, yb = next(iter(tr))
    print("train batch:", tuple(xb.shape), tuple(yb.shape))
    xb, yb = next(iter(va))
    print("val batch  :", tuple(xb.shape), tuple(yb.shape))


if __name__ == "__main__":
    _main()
