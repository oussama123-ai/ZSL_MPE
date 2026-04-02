"""
scripts/evaluate.py
====================
Zero-shot evaluation on benchmark datasets.

Usage
-----
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset unbc \
    --data_dir data/UNBC-McMaster \
    --config configs/default.yaml

python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset all \
    --unbc_dir data/UNBC-McMaster \
    --biovid_dir data/BioVid \
    --neonatal_dir data/neonatal \
    --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data.dataset          import SyntheticPainDataset, UnlabeledRealDataset
from models.pain_estimator import PainEstimator
from evaluation.evaluator  import Evaluator
from utils.logger          import setup_logging
from utils.checkpoint      import load_checkpoint
from torch.utils.data      import DataLoader

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate zero-shot pain estimator")
    p.add_argument("--config",       default="configs/default.yaml")
    p.add_argument("--checkpoint",   required=True)
    p.add_argument("--dataset",      default="unbc",
                   choices=["unbc", "biovid", "neonatal", "synthetic", "all"])
    p.add_argument("--data_dir",     default=None,
                   help="For single-dataset evaluation")
    p.add_argument("--unbc_dir",     default="data/UNBC-McMaster")
    p.add_argument("--biovid_dir",   default="data/BioVid")
    p.add_argument("--neonatal_dir", default="data/neonatal")
    p.add_argument("--synthetic_dir", default="data/synthetic")
    p.add_argument("--output",       default="results.json")
    p.add_argument("--device",       default=None)
    p.add_argument("--batch_size",   type=int, default=16)
    p.add_argument("--num_workers",  type=int, default=4)
    p.add_argument("--log_dir",      default="logs")
    p.add_argument("--bootstrap",    type=int, default=10_000)
    return p.parse_args()


def build_loader(data_dir: str, cfg: dict, batch_size: int,
                 num_workers: int) -> DataLoader:
    """Build a DataLoader from a dataset directory (uses SyntheticPainDataset as proxy)."""
    ds = SyntheticPainDataset(data_dir, cfg.get("data", {}), augment=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, pin_memory=True)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_dir)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Evaluating on device: %s", device)

    # Load model
    model = PainEstimator(cfg)
    load_checkpoint(model, path=args.checkpoint, device=device)
    model = model.to(device)

    eval_cfg = cfg.get("evaluation", {})
    evaluator = Evaluator(
        model,
        device=device,
        pain_threshold=eval_cfg.get("pain_threshold", 4.0),
        bootstrap_iters=args.bootstrap,
    )

    # Build loaders based on requested dataset(s)
    loaders: dict[str, DataLoader] = {}
    data_cfg = cfg.get("data", {})

    def _loader(data_dir: str) -> DataLoader:
        return build_loader(data_dir, cfg, args.batch_size, args.num_workers)

    if args.dataset == "all":
        loaders = {
            "UNBC-McMaster": _loader(args.unbc_dir),
            "BioVid":        _loader(args.biovid_dir),
            "Neonatal":      _loader(args.neonatal_dir),
        }
    elif args.dataset == "unbc":
        loaders = {"UNBC-McMaster": _loader(args.data_dir or args.unbc_dir)}
    elif args.dataset == "biovid":
        loaders = {"BioVid": _loader(args.data_dir or args.biovid_dir)}
    elif args.dataset == "neonatal":
        loaders = {"Neonatal": _loader(args.data_dir or args.neonatal_dir)}
    elif args.dataset == "synthetic":
        loaders = {"Synthetic": _loader(args.data_dir or args.synthetic_dir)}

    # Run evaluation
    results = evaluator.cross_dataset_evaluation(loaders)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        # Convert numpy types to Python native for JSON serialisation
        def _convert(o):
            if hasattr(o, "item"):
                return o.item()
            if isinstance(o, dict):
                return {k: _convert(v) for k, v in o.items()}
            if isinstance(o, list):
                return [_convert(v) for v in o]
            return o
        json.dump(_convert(results), f, indent=2)

    logger.info("Results saved to: %s", output_path)

    # Print summary
    for ds_name, metrics in results.items():
        print(f"\n{'='*50}")
        print(f"  {ds_name}")
        print(f"{'='*50}")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"  {k:<20s} {v:.4f}")
            else:
                print(f"  {k:<20s} {v}")


if __name__ == "__main__":
    main()
