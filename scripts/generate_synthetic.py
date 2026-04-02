"""
scripts/generate_synthetic.py
==============================
Entry point for generating 50,000 synthetic pain scenarios.

Usage
-----
python scripts/generate_synthetic.py \
    --n_samples 50000 \
    --output_dir data/synthetic \
    --config configs/default.yaml \
    --device cuda
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.synthetic_generator import SyntheticPainGenerator
from utils.logger import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic pain scenarios")
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--n_samples",  type=int,  default=50_000)
    p.add_argument("--output_dir", default="data/synthetic")
    p.add_argument("--device",     default="cpu",
                   help="'cuda' to use GPU for diffusion model")
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--log_dir",    default="logs")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_dir)

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    syn_cfg = cfg.get("synthetic", {})
    syn_cfg["seed"] = args.seed

    logger.info("Initialising SyntheticPainGenerator (n=%d, device=%s)",
                args.n_samples, args.device)

    generator = SyntheticPainGenerator(syn_cfg, device=args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = generator.generate(
        n=args.n_samples,
        output_dir=output_dir,
        show_progress=True,
    )

    # Save physiological signals and frames to disk
    frames_dir = output_dir / "frames"
    physio_dir = output_dir / "physio"
    frames_dir.mkdir(exist_ok=True)
    physio_dir.mkdir(exist_ok=True)

    logger.info("Saving %d samples to disk …", len(samples))
    for s in samples:
        np.save(frames_dir / f"{s.sample_id}.npy", s.frames)
        np.savez_compressed(
            physio_dir / f"{s.sample_id}.npz",
            hrv=s.hrv_ibi,
            eda=s.eda_signal,
            tremor=s.tremor_signal,
        )

    logger.info("Done. Synthetic dataset saved to: %s", output_dir)

    # Quick summary
    pains = np.array([s.pain_intensity for s in samples])
    logger.info("Pain distribution — mean: %.2f  std: %.2f  min: %.2f  max: %.2f",
                pains.mean(), pains.std(), pains.min(), pains.max())
    passed = sum(s.quality_passed for s in samples)
    logger.info("QC pass rate: %.1f%%", passed / len(samples) * 100)


if __name__ == "__main__":
    main()
