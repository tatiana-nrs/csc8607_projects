from __future__ import annotations

"""
Évaluation — à implémenter.

Doit exposer un main() exécutable via :
    python -m src.evaluate --config configs/config.yaml --checkpoint artifacts/best.ckpt

Exigences minimales :
- charger le modèle et le checkpoint
- calculer et afficher/consigner les métriques de test
"""

import argparse
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

from src.data_loading import get_dataloaders
from src.model import build_model
from src.train import get_device, eval_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    #load conf
    cfg: Dict[str, Any] = yaml.safe_load(open(args.config, "r"))

    device = get_device(cfg.get("train", {}))
    print(f"[INFO] device = {device}")

    _, _, test_loader, meta = get_dataloaders(cfg)
    num_classes = int(meta.get("num_classes", cfg["model"]["num_classes"]))

    model = build_model(cfg).to(device)

    #load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    #eval
    test_loss, test_acc, test_f1 = eval_epoch(
        model,
        test_loader,
        device,
        num_classes=num_classes
    )

    print("TEST RESULTS")
    print(f"Test loss     : {test_loss:.4f}")
    print(f"Test accuracy : {test_acc:.4f}")
    print(f"Test F1 macro : {test_f1:.4f}")

    # Optional TensorBoard logging
    runs_dir = Path(cfg["paths"]["runs_dir"])
    writer = SummaryWriter(log_dir=str(runs_dir / "test_eval"))
    writer.add_scalar("test/loss", test_loss)
    writer.add_scalar("test/accuracy", test_acc)
    writer.add_scalar("test/f1", test_f1)
    writer.close()


if __name__ == "__main__":
    main()
