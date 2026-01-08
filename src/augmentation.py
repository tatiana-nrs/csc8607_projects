from typing import Dict, Any, Optional
import torchvision.transforms as T

"""
Data augmentation

Signature imposée :
get_augmentation_transforms(config: dict) -> objet/transform callable (ou None)
"""

def get_augmentation_transforms(config: Dict[str, Any]) -> Optional[T.Compose]:
    """
    Retourne les transformations d'augmentation. À implémenter.
    """
    aug = config.get("augment", {}) or {}
    tfms = []

    # random_crop
    rc = aug.get("random_crop", None)
    if isinstance(rc, dict):
        size = tuple(rc.get("size"))  
        scale = tuple(rc.get("scale"))
        ratio = tuple(rc.get("ratio"))
        tfms.append(T.RandomResizedCrop(size=size, scale=scale, ratio=ratio, antialias=True))

    # flip
    if aug.get("random_flip", False):
        tfms.append(T.RandomHorizontalFlip(p=0.5))

    return T.Compose(tfms) if tfms else None
