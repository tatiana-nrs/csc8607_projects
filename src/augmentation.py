# src/augmentation.py
from typing import Dict, Any, Optional
import torchvision.transforms as T


def get_augmentation_transforms(config: Dict[str, Any]) -> Optional[T.Compose]:
    """
    Signature imposée:
      get_augmentation_transforms(config: dict) -> transform callable (ou None)

    Transformations ALÉATOIRES (train uniquement) selon config["augment"].
    """
    aug = config.get("augment", {}) or {}
    tfms = []

    # random_crop
    rc = aug.get("random_crop", None)
    if isinstance(rc, dict):
        size = tuple(rc.get("size"))  # requis dans ton YAML
        scale = tuple(rc.get("scale"))
        ratio = tuple(rc.get("ratio"))
        tfms.append(T.RandomResizedCrop(size=size, scale=scale, ratio=ratio, antialias=True))

    # flip
    if aug.get("random_flip", False):
        tfms.append(T.RandomHorizontalFlip(p=0.5))

    # color jitter
    cj = aug.get("color_jitter", None)
    if isinstance(cj, dict):
        tfms.append(T.ColorJitter(
            brightness=cj.get("brightness", 0.0),
            contrast=cj.get("contrast", 0.0),
            saturation=cj.get("saturation", 0.0),
            hue=cj.get("hue", 0.0),
        ))

    return T.Compose(tfms) if tfms else None
