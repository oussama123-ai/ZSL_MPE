"""
scripts/train.py
=================
Main training entry point for zero-shot multimodal pain estimation.

Usage
-----
# Single GPU
python scripts/train.py \
    --config configs/default.yaml \
    --synthetic_dir data/synthetic \
    --real_unlabeled_dir data/real_unlabeled \
    --output_dir checkpoints/run_001

# Multi-GPU (4× A100) via torchrun
torchrun --nproc_per_node=4 scripts/train.py \
    --config configs/default.yaml \
    --synthetic_dir data/synthetic \
    --real_unlabeled_dir data/real_unlabeled \
    --output_dir checkpoints/run_001
"""

from __future__ import annotations

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset        import SyntheticPainDataset, UnlabeledRealDataset
from models.pain_estimator import PainEstimator
from training.trainer    import Trainer
from utils.logger        import setup_logging
from utils.checkpoint    import load_checkpoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train zero-shot pain estimator")
    p.add_argument("--config",           default="configs/default.yaml")
    p.add_argument("--synthetic_dir",    default="data/synthetic")
    p.add_argument("--real_unlabeled_dir", default="data/real_unlabeled")
    p.add_argument("--output_dir",       default="checkpoints")
    p.add_argument("--log_dir",          default="logs")
    p.add_argument("--resume",           default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--device",           default=None,
                   help="Override device (e.g. 'cpu', 'cuda:0')")
    p.add_argument("--seed",             type=int, default=42)
    p.add_argument("--num_workers",      type=int, default=4)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def is_ddp() -> bool:
    return "LOCAL_RANK" in os.environ


def setup_ddp() -> tuple[int, int]:
    local_rank  = int(os.environ.get("LOCAL_RANK", 0))
    world_size  = int(os.environ.get("WORLD_SIZE", 1))
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


# ---------------------------------------------------------------------------
# DataLoader builders
# ---------------------------------------------------------------------------

def build_loaders(
    cfg: dict,
    synthetic_dir: str,
    real_dir: str,
    num_workers: int,
    world_size: int = 1,
    rank: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:

    data_cfg  = cfg.get("data", {})
    train_cfg = cfg.get("training", {})
    s1_cfg    = train_cfg.get("stage1", {})

    # ── Synthetic dataset ──────────────────────────────────────────────────
    synth_ds = SyntheticPainDataset(synthetic_dir, data_cfg, augment=True)

    synth_sampler = (DistributedSampler(synth_ds, num_replicas=world_size, rank=rank)
                     if world_size > 1 else None)

    # Stratified sampler (only for single-process; DDP uses DistributedSampler)
    if synth_sampler is None:
        strat_sampler = synth_ds.make_stratified_sampler(
            oversample_extreme=cfg["training"]["stratification"].get("oversample_extreme", 1.5)
        )
        synth_sampler = strat_sampler

    synth_loader = DataLoader(
        synth_ds,
        batch_size=s1_cfg.get("batch_size", 32),
        sampler=synth_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Real unlabeled dataset ─────────────────────────────────────────────
    real_ds = UnlabeledRealDataset(real_dir, data_cfg, strong_augment=True)

    real_sampler = (DistributedSampler(real_ds, num_replicas=world_size, rank=rank)
                    if world_size > 1 else None)

    real_loader = DataLoader(
        real_ds,
        batch_size=train_cfg.get("stage2", {}).get("batch_size_real", 16),
        sampler=real_sampler,
        shuffle=(real_sampler is None),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Validation (held-out 10% of synthetic) ────────────────────────────
    # For real benchmarks, replace this with UNBC-McMaster loader
    val_loader = None
    val_size   = max(int(0.10 * len(synth_ds)), 1)
    train_size = len(synth_ds) - val_size
    _, val_ds  = torch.utils.data.random_split(synth_ds, [train_size, val_size])
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False,
                            num_workers=num_workers, pin_memory=True)

    return synth_loader, real_loader, val_loader


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # DDP init
    ddp       = is_ddp()
    rank, world_size = (setup_ddp() if ddp else (0, 1))
    is_main   = (rank == 0)

    if is_main:
        setup_logging(args.log_dir)

    set_seed(args.seed + rank)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Device
    if args.device:
        device = args.device
    elif ddp:
        device = f"cuda:{rank}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    if is_main:
        logger.info("Device: %s | World size: %d", device, world_size)

    # Build model
    model = PainEstimator(cfg)
    if is_main:
        n_params = model.count_parameters()
        logger.info("Model parameters: %s", f"{n_params:,}")

    if args.resume:
        load_checkpoint(model, path=args.resume, device=device)

    model = model.to(device)
    if ddp:
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    # Build data loaders
    synth_loader, real_loader, val_loader = build_loaders(
        cfg,
        args.synthetic_dir,
        args.real_unlabeled_dir,
        args.num_workers,
        world_size,
        rank,
    )

    if is_main:
        logger.info("Synthetic samples: %d | Real unlabeled: %d",
                    len(synth_loader.dataset), len(real_loader.dataset))

    # Train
    raw_model = model.module if ddp else model
    trainer   = Trainer(
        cfg         = cfg,
        model       = raw_model,
        synth_loader= synth_loader,
        real_loader = real_loader,
        val_loader  = val_loader if is_main else None,
        device      = device,
        output_dir  = args.output_dir,
    )
    trainer.train()

    cleanup_ddp()


if __name__ == "__main__":
    main()
