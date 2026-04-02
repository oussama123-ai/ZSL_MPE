"""
Dataset Classes
===============
PyTorch Dataset wrappers for:
  - Synthetic pain data (labeled)
  - Unlabeled real-world data (for domain alignment)
  - Benchmark evaluation datasets (UNBC-McMaster, BioVid, Neonatal)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from data.augmentations import VideoAugmentor, PhysioAugmentor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper: Physiological feature extraction
# ---------------------------------------------------------------------------

def extract_hrv_features(ibi_ms: np.ndarray) -> np.ndarray:
    """Compute time-domain HRV features from IBI sequence."""
    diff  = np.diff(ibi_ms)
    sdnn  = np.std(ibi_ms)
    rmssd = np.sqrt(np.mean(diff ** 2))
    pnn50 = np.mean(np.abs(diff) > 50) * 100
    mean_hr = 60_000.0 / (np.mean(ibi_ms) + 1e-8)
    lf_hf = 0.5 + 1.5 * (1 - np.tanh(mean_hr / 120))
    return np.array([sdnn, rmssd, pnn50, mean_hr, lf_hf], dtype=np.float32)


def pad_or_trim(arr: np.ndarray, target_len: int) -> np.ndarray:
    """Pad with zeros or trim a 1-D signal to target_len."""
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)))


# ---------------------------------------------------------------------------
# Demographic encoding
# ---------------------------------------------------------------------------

AGE_MAP  = {"neonate": 0, "child": 1, "adult": 2, "elderly": 3}
ETH_MAP  = {"caucasian": 0, "african_american": 1, "asian": 2, "hispanic": 3, "other": 4}
SEX_MAP  = {"female": 0, "male": 1}

CLINICAL_SETTINGS = ["post_surgical", "procedural", "chronic", "trauma", "disease"]
PAIN_TYPES        = ["acute_nociceptive", "chronic", "neuropathic", "procedural", "inflammatory"]


def encode_context(age_group: str, ethnicity: str, sex: str,
                   clinical_setting: str, pain_type: str) -> Dict[str, torch.Tensor]:
    """Return integer indices + multi-hot clinical vectors."""
    age = torch.tensor(AGE_MAP.get(age_group, 2), dtype=torch.long)
    eth = torch.tensor(ETH_MAP.get(ethnicity, 0), dtype=torch.long)
    s   = torch.tensor(SEX_MAP.get(sex, 0), dtype=torch.long)

    v_setting = torch.zeros(len(CLINICAL_SETTINGS))
    if clinical_setting in CLINICAL_SETTINGS:
        v_setting[CLINICAL_SETTINGS.index(clinical_setting)] = 1.0

    v_type = torch.zeros(len(PAIN_TYPES))
    if pain_type in PAIN_TYPES:
        v_type[PAIN_TYPES.index(pain_type)] = 1.0

    return {
        "age":              age,
        "ethnicity":        eth,
        "sex":              s,
        "clinical_setting": v_setting,
        "pain_type":        v_type,
    }


# ---------------------------------------------------------------------------
# Synthetic Dataset
# ---------------------------------------------------------------------------

class SyntheticPainDataset(Dataset):
    """
    Loads pre-generated synthetic pain scenarios.

    Expected directory structure:
        synthetic_dir/
          metadata_final.json     ← pain labels & demographics
          frames/
            synth_000000.npy      ← (T, H, W, 3) uint8
          physio/
            synth_000000.npz      ← {"hrv": ..., "eda": ..., "tremor": ...}

    Falls back to on-the-fly generation if files don't exist.
    """

    def __init__(
        self,
        synthetic_dir: Union[str, Path],
        cfg: dict,
        augment: bool = True,
        generator=None,         # SyntheticPainGenerator, for on-the-fly mode
    ):
        super().__init__()
        self.root    = Path(synthetic_dir)
        self.cfg     = cfg
        self.augment = augment
        self.generator = generator

        self.video_aug = VideoAugmentor(cfg, strong=False)
        self.physio_aug = PhysioAugmentor(cfg)

        # Load metadata
        meta_path = self.root / "metadata_final.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
            logger.info("Loaded %d synthetic samples from %s", len(self.meta), self.root)
        elif generator is not None:
            logger.warning("metadata_final.json not found; using on-the-fly generation.")
            self.meta = None
        else:
            raise FileNotFoundError(f"No metadata at {meta_path} and no generator provided.")

        self._n_samples = len(self.meta) if self.meta else cfg.get("n_samples", 50_000)
        self._physio_len = int(cfg.get("physio_window_seconds", 120)
                               * cfg.get("eda_sample_rate", 100))
        self._hrv_len    = 200  # ~120 s at ~60 bpm

    def __len__(self) -> int:
        return self._n_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if self.meta is not None:
            record = self.meta[idx]
            frames, hrv, eda, tremor = self._load_arrays(record["id"])
            pain  = float(record["pain"])
            context = encode_context(
                record["age_group"], record["ethnicity"], record["sex"],
                record["clinical_setting"], record["pain_type"],
            )
            au_vec = torch.tensor(
                [record["au_activations"].get(au, 0.0) for au in
                 ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]],
                dtype=torch.float32,
            )
        else:
            # On-the-fly
            sample = self.generator.generate_one(f"synth_{idx:06d}")
            frames = sample.frames
            hrv, eda, tremor = sample.hrv_ibi, sample.eda_signal, sample.tremor_signal
            pain  = sample.pain_intensity
            context = encode_context(
                sample.age_group, sample.ethnicity, sample.sex,
                sample.clinical_setting, sample.pain_type,
            )
            au_vec = torch.tensor(
                [sample.au_activations.get(au, 0.0) for au in
                 ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]],
                dtype=torch.float32,
            )

        # Augment
        if self.augment:
            frames = self.video_aug(frames)
            hrv, eda, tremor = self.physio_aug(hrv, eda, tremor)

        video  = self._frames_to_tensor(frames)
        hrv_t  = torch.from_numpy(pad_or_trim(hrv,    self._hrv_len)).unsqueeze(0)
        eda_t  = torch.from_numpy(pad_or_trim(eda,    self._physio_len)).unsqueeze(0)
        trem_t = torch.from_numpy(pad_or_trim(tremor, self._physio_len)).unsqueeze(0)
        pain_t = torch.tensor(pain / 10.0, dtype=torch.float32)  # normalised [0,1]

        return {
            "video":            video,
            "hrv":              hrv_t,
            "eda":              eda_t,
            "tremor":           trem_t,
            "pain":             pain_t,
            "au":               au_vec,
            "age":              context["age"],
            "ethnicity":        context["ethnicity"],
            "sex":              context["sex"],
            "clinical_setting": context["clinical_setting"],
            "pain_type":        context["pain_type"],
            "is_synthetic":     torch.tensor(1, dtype=torch.long),
        }

    def _load_arrays(self, sample_id: str
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_path  = self.root / "frames"  / f"{sample_id}.npy"
        physio_path = self.root / "physio"  / f"{sample_id}.npz"

        if frame_path.exists():
            frames = np.load(frame_path)
        else:
            T, H, W = 30, 224, 224
            frames = np.random.randint(100, 200, (T, H, W, 3), dtype=np.uint8)

        if physio_path.exists():
            d = np.load(physio_path)
            hrv, eda, tremor = d["hrv"], d["eda"], d["tremor"]
        else:
            hrv    = np.random.normal(65, 10, 200).astype(np.float32)
            eda    = np.random.uniform(2, 8, 12_000).astype(np.float32)
            tremor = np.random.normal(0, 0.1, 12_000).astype(np.float32)

        return frames, hrv, eda, tremor

    @staticmethod
    def _frames_to_tensor(frames: np.ndarray) -> torch.Tensor:
        """(T, H, W, 3) uint8 → (T, 3, H, W) float32 [0,1]."""
        t = torch.from_numpy(frames.astype(np.float32) / 255.0)
        return t.permute(0, 3, 1, 2)

    # -- Stratified sampler for pain intensity balance -----------------------

    def make_stratified_sampler(self, n_bins: int = 5,
                                oversample_extreme: float = 1.5) -> WeightedRandomSampler:
        """Returns a WeightedRandomSampler balancing pain intensity bins."""
        if self.meta is None:
            return None
        pains  = np.array([r["pain"] for r in self.meta], dtype=np.float32)
        bins   = np.linspace(0, 10, n_bins + 1)
        labels = np.digitize(pains, bins) - 1
        labels = np.clip(labels, 0, n_bins - 1)

        bin_counts = np.bincount(labels, minlength=n_bins).astype(float)
        bin_counts[bin_counts == 0] = 1
        weights = 1.0 / bin_counts[labels]
        # Oversample extreme pain (bin 4 → 8–10)
        weights[labels == n_bins - 1] *= oversample_extreme

        return WeightedRandomSampler(
            torch.from_numpy(weights).double(),
            num_samples=len(pains),
            replacement=True,
        )


# ---------------------------------------------------------------------------
# Unlabeled Real Dataset (for domain alignment)
# ---------------------------------------------------------------------------

class UnlabeledRealDataset(Dataset):
    """
    Unlabeled real facial / physiological recordings used in Stage 2–3
    for domain alignment. No pain labels used.

    Expected structure:
        real_dir/
          metadata.json           ← list of {id, age_group, sex, ...}
          frames/ *.npy
          physio/ *.npz
    """

    def __init__(
        self,
        real_dir: Union[str, Path],
        cfg: dict,
        strong_augment: bool = True,
    ):
        super().__init__()
        self.root = Path(real_dir)
        self.cfg  = cfg
        self.video_aug  = VideoAugmentor(cfg, strong=strong_augment)
        self.physio_aug = PhysioAugmentor(cfg)

        meta_path = self.root / "metadata.json"
        if meta_path.exists():
            with open(meta_path) as f:
                self.meta = json.load(f)
        else:
            # Graceful fallback: scan for numpy files
            frames_dir = self.root / "frames"
            if frames_dir.exists():
                self.meta = [{"id": p.stem} for p in sorted(frames_dir.glob("*.npy"))]
            else:
                logger.warning("Real unlabeled data directory empty: %s", real_dir)
                self.meta = []

        self._physio_len = int(cfg.get("physio_window_seconds", 120)
                               * cfg.get("eda_sample_rate", 100))
        self._hrv_len = 200

    def __len__(self) -> int:
        return max(len(self.meta), 1)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if not self.meta:
            return self._dummy()

        record = self.meta[idx % len(self.meta)]
        frames, hrv, eda, tremor = self._load_arrays(record["id"])

        frames_aug = self.video_aug(frames)
        hrv_a, eda_a, trem_a = self.physio_aug(hrv, eda, tremor)

        video  = SyntheticPainDataset._frames_to_tensor(frames_aug)
        hrv_t  = torch.from_numpy(pad_or_trim(hrv_a,   self._hrv_len)).unsqueeze(0)
        eda_t  = torch.from_numpy(pad_or_trim(eda_a,   self._physio_len)).unsqueeze(0)
        trem_t = torch.from_numpy(pad_or_trim(trem_a,  self._physio_len)).unsqueeze(0)

        context = encode_context(
            record.get("age_group", "adult"),
            record.get("ethnicity", "caucasian"),
            record.get("sex", "female"),
            record.get("clinical_setting", "procedural"),
            record.get("pain_type", "acute_nociceptive"),
        )

        return {
            "video":            video,
            "hrv":              hrv_t,
            "eda":              eda_t,
            "tremor":           trem_t,
            "age":              context["age"],
            "ethnicity":        context["ethnicity"],
            "sex":              context["sex"],
            "clinical_setting": context["clinical_setting"],
            "pain_type":        context["pain_type"],
            "is_synthetic":     torch.tensor(0, dtype=torch.long),
        }

    def _load_arrays(self, sid: str
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        frame_path  = self.root / "frames"  / f"{sid}.npy"
        physio_path = self.root / "physio"  / f"{sid}.npz"

        frames = (np.load(frame_path) if frame_path.exists()
                  else np.random.randint(100, 200, (30, 224, 224, 3), dtype=np.uint8))
        if physio_path.exists():
            d = np.load(physio_path)
            hrv, eda, tremor = d["hrv"], d["eda"], d["tremor"]
        else:
            hrv    = np.random.normal(65, 10, 200).astype(np.float32)
            eda    = np.random.uniform(2, 8, 12_000).astype(np.float32)
            tremor = np.random.normal(0, 0.1, 12_000).astype(np.float32)
        return frames, hrv, eda, tremor

    def _dummy(self) -> Dict[str, torch.Tensor]:
        T, H, W = 30, 224, 224
        return {
            "video":            torch.zeros(T, 3, H, W),
            "hrv":              torch.zeros(1, self._hrv_len),
            "eda":              torch.zeros(1, self._physio_len),
            "tremor":           torch.zeros(1, self._physio_len),
            "age":              torch.tensor(2, dtype=torch.long),
            "ethnicity":        torch.tensor(0, dtype=torch.long),
            "sex":              torch.tensor(0, dtype=torch.long),
            "clinical_setting": torch.zeros(5),
            "pain_type":        torch.zeros(5),
            "is_synthetic":     torch.tensor(0, dtype=torch.long),
        }
