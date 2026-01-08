from typing import Dict, Any
import torchvision.transforms as T
from PIL import Image

"""
Pré-traitements.

Signature imposée :
get_preprocess_transforms(config: dict) -> objet/transform callable
"""

class ToRGB:
    """Force l'image en RGB"""
    def __call__(self, img):
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img


def get_preprocess_transforms(config: Dict[str, Any]):
    """
    Retourne les transformations de pré-traitement. À implémenter
    Transformations invariantes (train/val/test):
      ToRGB -> Resize(optional) -> ToTensor -> Normalize(mean/std depuis YAML)
    """
    pp = config.get("preprocess", {}) or {}

    # resize depuis YAML
    resize = pp.get("resize", None)
    norm = pp.get("normalize", {}) or {}
    mean = norm.get("mean", None)
    std = norm.get("std", None)

    if mean is None or std is None:
        raise ValueError("configs/config.yaml: preprocess.normalize.mean et preprocess.normalize.std sont requis.")

    tfms = [ToRGB()]

    if resize is not None:
        tfms.append(T.Resize(tuple(resize), antialias=True))

    tfms += [
        T.ToTensor(),
        T.Normalize(mean=mean, std=std),
    ]

    return T.Compose(tfms)
