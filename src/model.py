# src/model.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn


class DepthwiseSeparableBlock(nn.Module):
    """
    Bloc imposé :
    Depthwise 3x3 (groups=Cin) -> BN -> ReLU -> Pointwise 1x1 -> BN -> ReLU
    """
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()

        self.depthwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,      # conserve Cin
            kernel_size=3,
            padding=1,
            groups=in_ch,            # depthwise
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)

        self.pointwise = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,     # mélange + ajuste #canaux
            kernel_size=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu2 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout(p=dropout) if dropout and dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.depthwise(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.pointwise(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.drop(x)
        return x


def _read_B_and_width(config: Dict[str, Any]) -> Tuple[List[int], float]:
    """
    Lit B=(B1,B2,B3) et width depuis la config, sans hardcoder.
    Priorité:
      1) config["model"]["B"] / config["model"]["width"]
      2) config["hparams"]["B"] / config["hparams"]["width"] SI ce sont déjà des scalaires (pas une grille)
      3) fallback (1,1,1) et 1.0
    """
    model_cfg = config.get("model", {}) or {}
    h_cfg = config.get("hparams", {}) or {}

    B = model_cfg.get("B", None)
    width = model_cfg.get("width", None)

    # fallback possible si grid_search écrase hparams avec une valeur scalaire
    if B is None:
        B_candidate = h_cfg.get("B", None)
        if isinstance(B_candidate, list) and len(B_candidate) == 3 and all(isinstance(x, int) for x in B_candidate):
            B = B_candidate

    if width is None:
        w_candidate = h_cfg.get("width", None)
        if isinstance(w_candidate, (int, float)):
            width = float(w_candidate)

    if B is None:
        B = [1, 1, 1]
    if width is None:
        width = 1.0

    # sécurité
    if not (isinstance(B, list) and len(B) == 3 and all(isinstance(x, int) for x in B)):
        raise ValueError(f"B invalide (attendu [B1,B2,B3]) mais obtenu: {B}")
    if width not in (0.75, 1.0):
        raise ValueError(f"width doit être 0.75 ou 1.0, obtenu: {width}")

    return B, float(width)


class DSConvNet(nn.Module):
    """
    Modèle imposé :
      - 3 stages
      - après stage1 et stage2: MaxPool2d(2)
      - après stage3: GlobalAveragePooling
      - tête: Linear(C_final -> num_classes)
    """
    def __init__(self, num_classes: int, B: List[int], width: float, dropout: float = 0.0):
        super().__init__()
        B1, B2, B3 = B

        c1 = int(round(64 * width))
        c2 = int(round(128 * width))
        c3 = int(round(256 * width))

        # Stage 1 : entrée RGB => 3 canaux
        stage1 = []
        in_ch = 3
        for i in range(B1):
            out_ch = c1  # chaque bloc du stage produit c1
            stage1.append(DepthwiseSeparableBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        self.stage1 = nn.Sequential(*stage1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 2
        stage2 = []
        for i in range(B2):
            out_ch = c2
            stage2.append(DepthwiseSeparableBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        self.stage2 = nn.Sequential(*stage2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Stage 3
        stage3 = []
        for i in range(B3):
            out_ch = c3
            stage3.append(DepthwiseSeparableBlock(in_ch, out_ch, dropout=dropout))
            in_ch = out_ch
        self.stage3 = nn.Sequential(*stage3)

        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(in_features=c3, out_features=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stage1(x)
        x = self.pool1(x)

        x = self.stage2(x)
        x = self.pool2(x)

        x = self.stage3(x)
        x = self.gap(x)              # (B, C, 1, 1)
        x = torch.flatten(x, 1)      # (B, C)
        logits = self.classifier(x)  # (B, num_classes)
        return logits


def build_model(config: Dict[str, Any]) -> nn.Module:
    """
    Fonction imposée par le dépôt.
    Doit retourner un modèle PyTorch qui sort des logits (B, num_classes).
    """
    model_cfg = config.get("model", {}) or {}
    num_classes = int(model_cfg.get("num_classes", 200))
    dropout = float(model_cfg.get("dropout", 0.0))

    B, width = _read_B_and_width(config)

    if model_cfg.get("type", "dsconvnet") != "dsconvnet":
        raise ValueError(f"model.type non supporté ici: {model_cfg.get('type')}")

    return DSConvNet(num_classes=num_classes, B=B, width=width, dropout=dropout)
