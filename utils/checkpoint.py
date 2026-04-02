"""Checkpoint utilities."""
from __future__ import annotations
from pathlib import Path
import torch
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(model, optimizer, epoch: int, val_mae: float,
                    path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch":     epoch,
        "val_mae":   val_mae,
        "model":     model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }, path)
    logger.info("Checkpoint saved → %s  (epoch %d, val_MAE=%.4f)", path, epoch, val_mae)


def load_checkpoint(model, optimizer=None, path: str = "best_model.pt",
                    device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    logger.info("Loaded checkpoint from %s  (epoch %d, val_MAE=%.4f)",
                path, ckpt.get("epoch", -1), ckpt.get("val_mae", -1))
    return ckpt
