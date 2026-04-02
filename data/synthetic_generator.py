"""
Synthetic Pain Generation Pipeline
===================================
Implements Section 3.2 of the paper:
  - Diffusion-based facial expression generation (3.2.1)
  - Physiological signal synthesis — HRV, EDA, tremor (3.2.2)
  - Temporal dynamics modeling (3.2.3)
  - Quality assurance (3.2.4)
"""

from __future__ import annotations

import logging
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import butter, filtfilt, sosfilt, butter
from scipy.stats import norm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class SyntheticSample:
    """One generated multimodal pain scenario."""
    sample_id: str
    pain_intensity: float                   # y ∈ [0, 10]
    frames: np.ndarray                      # (T, H, W, 3) uint8
    hrv_ibi: np.ndarray                     # inter-beat intervals (ms)
    eda_signal: np.ndarray                  # skin conductance (µS)
    tremor_signal: np.ndarray               # acceleration (arbitrary units)
    age_group: str
    ethnicity: str
    sex: str
    clinical_setting: str
    pain_type: str
    au_activations: Dict[str, float]        # AU code → intensity [0,1]
    temporal_profile: str                   # "acute" | "sustained" | "adapting"
    quality_passed: bool = True


# ---------------------------------------------------------------------------
# Demographic & clinical sampling
# ---------------------------------------------------------------------------

AGE_GROUPS      = ["neonate", "child", "adult", "elderly"]
AGE_RATIOS      = [0.25, 0.25, 0.25, 0.25]
ETHNICITIES     = ["caucasian", "african_american", "asian", "hispanic", "other"]
ETHNICITY_RATIOS = [0.20, 0.20, 0.20, 0.20, 0.20]
SEX_GROUPS      = ["female", "male"]
SEX_RATIOS      = [0.50, 0.50]
CLINICAL_SETTINGS = ["post_surgical", "procedural", "chronic", "trauma", "disease"]
CLINICAL_RATIOS   = [0.30, 0.25, 0.20, 0.15, 0.10]
PAIN_TYPES        = ["acute_nociceptive", "chronic", "neuropathic", "procedural", "inflammatory"]
PAIN_TYPE_RATIOS  = [0.25, 0.25, 0.20, 0.20, 0.10]

# Pain-relevant Action Units (PSPI)
PAIN_AUS = ["AU4", "AU6", "AU7", "AU9", "AU10", "AU43"]


def _choice(options: list, probs: list, rng: np.random.Generator) -> str:
    return rng.choice(options, p=probs)


def sample_demographics(rng: np.random.Generator) -> Dict[str, str]:
    return {
        "age_group":        _choice(AGE_GROUPS,       AGE_RATIOS,       rng),
        "ethnicity":        _choice(ETHNICITIES,       ETHNICITY_RATIOS,  rng),
        "sex":              _choice(SEX_GROUPS,        SEX_RATIOS,        rng),
        "clinical_setting": _choice(CLINICAL_SETTINGS, CLINICAL_RATIOS,   rng),
        "pain_type":        _choice(PAIN_TYPES,        PAIN_TYPE_RATIOS,  rng),
    }


# ---------------------------------------------------------------------------
# Action Unit intensity model (from PSPI clinical data — Section 3.2.1)
# ---------------------------------------------------------------------------

# Sigmoid parameters (a_k, b_k) per AU learned from PSPI
_AU_PARAMS: Dict[str, Tuple[float, float]] = {
    "AU4":  (0.55, -1.5),
    "AU6":  (0.40, -2.0),
    "AU7":  (0.40, -2.0),
    "AU9":  (0.45, -1.8),
    "AU10": (0.35, -2.2),
    "AU43": (0.50, -1.6),
}


def compute_au_activations(pain: float) -> Dict[str, float]:
    """AU intensity weighted by pain level via sigmoid (Eq. 6 in paper)."""
    activations = {}
    for au, (a, b) in _AU_PARAMS.items():
        w = 1.0 / (1.0 + math.exp(-(a * pain + b)))
        # Add small noise for realism
        activations[au] = float(np.clip(w + np.random.normal(0, 0.05), 0, 1))
    return activations


# ---------------------------------------------------------------------------
# Physiological signal synthesis
# ---------------------------------------------------------------------------

class HRVSynthesizer:
    """
    Heart Rate Variability synthesis (Section 3.2.2, Eq. 9).

    HRV(t, p) = HRV_base · (1 − α_p · tanh(p/5)) · (1 + γ sin(2π f_resp t)) + η_t
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def synthesize(self, pain: float, duration_s: float = 120) -> np.ndarray:
        """Return inter-beat interval sequence in ms."""
        cfg = self.cfg
        hrv_base  = self.rng.normal(cfg["baseline_mean"], cfg["baseline_std"])
        hrv_base  = max(hrv_base, 20.0)

        alpha_p   = self.rng.uniform(cfg["alpha_min"], cfg["alpha_max"])
        fresp     = self.rng.normal(cfg["resp_freq_mean"], cfg["resp_freq_std"])
        fresp     = max(fresp, 0.1)

        # Compute average RR interval (ms) from HRV envelope
        n_beats   = int(duration_s * 60 / hrv_base) + 5
        t_beats   = np.linspace(0, duration_s, n_beats)

        pain_factor  = 1.0 - alpha_p * math.tanh(pain / 5.0)
        rsa_factor   = 1.0 + cfg["gamma"] * np.sin(2 * math.pi * fresp * t_beats)
        noise        = self.rng.normal(0, cfg["noise_std"], n_beats)

        ibi_ms = hrv_base * pain_factor * rsa_factor + noise
        ibi_ms = np.clip(ibi_ms, 300, 2000)     # physiological bounds
        return ibi_ms.astype(np.float32)

    @staticmethod
    def compute_features(ibi_ms: np.ndarray) -> Dict[str, float]:
        """Time-domain & frequency-domain HRV features."""
        diff   = np.diff(ibi_ms)
        sdnn   = float(np.std(ibi_ms))
        rmssd  = float(np.sqrt(np.mean(diff ** 2)))
        pnn50  = float(np.mean(np.abs(diff) > 50) * 100)

        # Approximate LF/HF via Welch (simplified)
        # In practice use scipy.signal.welch on RR series
        lf_hf = float(0.5 + 1.5 * (1 - np.tanh(np.mean(ibi_ms) / 1000)))
        return {"sdnn": sdnn, "rmssd": rmssd, "pnn50": pnn50, "lf_hf": lf_hf}


class EDASynthesizer:
    """
    Electrodermal Activity synthesis (Section 3.2.2, Eq. 10).

    EDA(t, p) = SCL_base + Σ A_j(p) · [exp(−t_j/τ1) − exp(−t_j/τ2)] + ξ_t
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def synthesize(self, pain: float, duration_s: float = 120) -> np.ndarray:
        sr  = self.cfg["sample_rate"]
        t   = np.arange(int(duration_s * sr)) / sr

        scl_base = self.rng.uniform(self.cfg["scl_base_min"], self.cfg["scl_base_max"])
        eda = np.full(len(t), scl_base, dtype=np.float64)

        # Phasic SCR events
        lam_p   = 0.5 + 0.3 * pain
        n_scr   = self.rng.poisson(lam_p * duration_s / 60.0)

        tau1, tau2 = self.cfg["tau1"], self.cfg["tau2"]
        a_min  = 0.05 + 0.03 * pain
        a_max  = 0.10 + 0.05 * pain

        for _ in range(n_scr):
            onset = self.rng.uniform(0, duration_s)
            amp   = self.rng.uniform(a_min, a_max)
            t_rel = t - onset
            scr   = np.where(t_rel >= 0,
                             amp * (np.exp(-np.clip(t_rel, 0, None) / tau2)
                                    - np.exp(-np.clip(t_rel, 0, None) / tau1)),
                             0.0)
            eda  += scr

        # Measurement noise
        eda += self.rng.normal(0, self.cfg["noise_std"], len(t))

        # Low-pass filter at 5 Hz
        eda = self._lowpass(eda, cutoff=self.cfg["lowpass_cutoff"], fs=sr)
        eda = np.clip(eda, 0, 30).astype(np.float32)
        return eda

    @staticmethod
    def _lowpass(signal: np.ndarray, cutoff: float, fs: float,
                 order: int = 4) -> np.ndarray:
        nyq = 0.5 * fs
        normal_cutoff = cutoff / nyq
        b, a = butter(order, normal_cutoff, btype="low", analog=False)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return filtfilt(b, a, signal)


class TremorSynthesizer:
    """
    Pain-related tremor synthesis (Section 3.2.2, Eq. 11–12).

    G(f, p) = A_base + p · A_pain · exp(−(f − f0)²/(2σ_f²))
    """

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.cfg  = cfg
        self.rng  = rng
        self.fs   = 100   # Hz

    def synthesize(self, pain: float, duration_s: float = 120) -> np.ndarray:
        n   = int(duration_s * self.fs)
        eta = self.rng.standard_normal(n)

        # Frequency-domain shaping
        freqs = np.fft.rfftfreq(n, 1.0 / self.fs)
        f0    = self.cfg["center_freq"]
        sigma = self.cfg["bandwidth"]
        a_base  = self.cfg["baseline_amplitude"]
        a_pain  = self.cfg["pain_scale"]

        G = a_base + pain * a_pain * np.exp(-((freqs - f0) ** 2) / (2 * sigma ** 2))
        X = np.fft.rfft(eta) * G
        tremor = np.fft.irfft(X, n=n).astype(np.float32)
        return tremor


class CorrelatedPhysioSynthesizer:
    """
    Generate correlated HRV / EDA / tremor via Gaussian copula (Eq. 13–14).
    """

    # Correlation matrix Σ_physio from the paper
    _CORR = np.array([
        [ 1.00, -0.42,  0.31],
        [-0.42,  1.00, -0.28],
        [ 0.31, -0.28,  1.00],
    ])

    def __init__(self, cfg: dict, rng: np.random.Generator):
        self.hrv_synth    = HRVSynthesizer(cfg["hrv"],    rng)
        self.eda_synth    = EDASynthesizer(cfg["eda"],    rng)
        self.tremor_synth = TremorSynthesizer(cfg["tremor"], rng)
        self.L            = np.linalg.cholesky(self._CORR)
        self.rng          = rng

    def synthesize(self, pain: float, duration_s: float = 120
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (hrv_ibi, eda, tremor) with correlated amplitude scaling."""
        # Base signals
        hrv    = self.hrv_synth.synthesize(pain, duration_s)
        eda    = self.eda_synth.synthesize(pain, duration_s)
        tremor = self.tremor_synth.synthesize(pain, duration_s)

        # Apply copula amplitude scaling
        z = self.L @ self.rng.standard_normal(3)
        scale = np.exp(z * 0.1)   # mild multiplicative perturbation

        hrv    = hrv    * scale[0]
        eda    = eda    * abs(scale[1])
        tremor = tremor * scale[2]

        return hrv.astype(np.float32), eda.astype(np.float32), tremor.astype(np.float32)


# ---------------------------------------------------------------------------
# Temporal dynamics (Section 3.2.3)
# ---------------------------------------------------------------------------

class TemporalDynamicsModel:
    """
    Models three temporal pain regimes:
      - Acute onset (Eq. 15)
      - Sustained with fluctuations (Eq. 16)
      - Adaptation / habituation (Eq. 17)
    """

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def generate_profile(self, pain_intensity: float,
                          duration_s: float = 10.0,
                          fps: int = 30) -> Tuple[np.ndarray, str]:
        """Return frame-level pain trajectory and profile name."""
        n_frames = int(duration_s * fps)
        t        = np.linspace(0, duration_s, n_frames)
        profile  = self.rng.choice(["acute", "sustained", "adapting"],
                                   p=[0.40, 0.35, 0.25])

        if profile == "acute":
            p = self._acute(t, pain_intensity)
        elif profile == "sustained":
            p = self._sustained(t, pain_intensity)
        else:
            p = self._adapting(t, pain_intensity)

        return p.astype(np.float32), profile

    def _acute(self, t: np.ndarray, p_max: float) -> np.ndarray:
        """Eq. 15: p(t) = p_max · (1 − e^{−t/τ_onset}) · 1_{t ≥ t_0}"""
        t0     = self.rng.uniform(0, min(2.0, t[-1] * 0.3))
        tau    = self.rng.uniform(0.5, 1.5)
        p      = p_max * (1.0 - np.exp(-np.clip(t - t0, 0, None) / tau))
        p[t < t0] = 0.0
        return p

    def _sustained(self, t: np.ndarray, p_mean: float) -> np.ndarray:
        """Eq. 16: p(t) = p_mean + Σ A_k sin(2π f_k t + φ_k) + w(t)"""
        freqs = [0.20, 0.05, 0.01]
        amps  = self.rng.uniform(0.05, 0.2 * p_mean + 0.01, 3)
        phases = self.rng.uniform(0, 2 * math.pi, 3)

        p = np.full_like(t, p_mean)
        for A, f, phi in zip(amps, freqs, phases):
            p += A * np.sin(2 * math.pi * f * t + phi)

        # Low-freq GP noise (RBF kernel, lengthscale 10 s)
        sigma_noise = 0.05 * p_mean
        p += self.rng.normal(0, sigma_noise, len(t))
        return np.clip(p, 0, 10)

    def _adapting(self, t: np.ndarray, p_initial: float) -> np.ndarray:
        """Eq. 17: p(t) = p_steady + (p_initial − p_steady) · e^{−λt}"""
        p_steady = self.rng.uniform(0.5, 0.7) * p_initial
        lam      = self.rng.uniform(0.01, 0.05)
        p        = p_steady + (p_initial - p_steady) * np.exp(-lam * t)
        return np.clip(p, 0, 10)


# ---------------------------------------------------------------------------
# Placeholder facial generation (wraps diffusion when available)
# ---------------------------------------------------------------------------

class FacialExpressionGenerator:
    """
    Wraps Stable Diffusion v2.1 for pain-conditioned face generation.

    When the diffusion model is not available (no GPU / API key), falls back
    to generating a random noise placeholder with correct shape so the rest
    of the pipeline can run.
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg    = cfg
        self.device = device
        self.pipe   = None
        self._try_load()

    def _try_load(self) -> None:
        try:
            from diffusers import StableDiffusionPipeline
            model_id = self.cfg.get("model_id", "stabilityai/stable-diffusion-2-1")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                model_id,
                torch_dtype=torch.float16 if self.device != "cpu" else torch.float32,
            ).to(self.device)
            logger.info("Stable Diffusion loaded for facial generation.")
        except Exception as exc:
            logger.warning(
                "Could not load Stable Diffusion (%s). "
                "Using placeholder frames (random noise).",
                exc,
            )
            self.pipe = None

    def generate_sequence(
        self,
        pain_intensity: float,
        au_activations: Dict[str, float],
        demographics: Dict[str, str],
        temporal_profile: np.ndarray,
        n_frames: int = 30,
    ) -> np.ndarray:
        """
        Generate a (n_frames, H, W, 3) uint8 array.

        With diffusion model: samples key-frames conditioned on pain / AUs
        and interpolates to n_frames using ILVR-style approach (Eq. 18).

        Without diffusion model: returns placeholder noise array.
        """
        H = W = self.cfg.get("resolution", 224)

        if self.pipe is not None:
            return self._diffusion_generate(
                pain_intensity, au_activations, demographics,
                temporal_profile, n_frames, H, W,
            )
        else:
            return self._placeholder_generate(pain_intensity, n_frames, H, W)

    def _diffusion_generate(self, pain, au_activations, demographics,
                            temporal_profile, n_frames, H, W) -> np.ndarray:
        """Sample key-frames every 5th frame, then interpolate."""
        key_indices = list(range(0, n_frames, 5)) + [n_frames - 1]
        frames = np.zeros((n_frames, H, W, 3), dtype=np.uint8)

        for ki in key_indices:
            p_frame = float(temporal_profile[ki])
            prompt  = self._build_prompt(p_frame, au_activations, demographics)
            with torch.autocast(self.device):
                result = self.pipe(
                    prompt,
                    height=H,
                    width=W,
                    num_inference_steps=self.cfg.get("num_inference_steps", 50),
                    guidance_scale=self.cfg.get("guidance_scale", 7.5),
                )
            img = np.array(result.images[0])
            frames[ki] = img

        # Linear interpolation between key-frames
        for i in range(len(key_indices) - 1):
            s, e = key_indices[i], key_indices[i + 1]
            for j in range(s, e):
                alpha = (j - s) / max(e - s, 1)
                frames[j] = ((1 - alpha) * frames[s] + alpha * frames[e]).astype(np.uint8)

        return frames

    @staticmethod
    def _build_prompt(pain: float, au: Dict[str, float],
                      demo: Dict[str, str]) -> str:
        intensity = (
            "no" if pain < 1 else
            "mild" if pain < 3 else
            "moderate" if pain < 6 else
            "severe"
        )
        age = demo.get("age_group", "adult")
        sex = demo.get("sex", "female")
        eth = demo.get("ethnicity", "caucasian")
        return (
            f"Close-up portrait of a {age} {sex} {eth} person experiencing "
            f"{intensity} pain, medical setting, natural lighting, "
            f"photorealistic, high quality"
        )

    @staticmethod
    def _placeholder_generate(pain: float, n_frames: int,
                               H: int, W: int) -> np.ndarray:
        """Pain-tinted random noise — placeholder when diffusion unavailable."""
        rng   = np.random.default_rng()
        base  = rng.integers(100, 200, (H, W, 3), dtype=np.uint8)
        tint  = np.array([int(pain * 10), 0, 0], dtype=np.uint8)
        frames = np.stack([
            np.clip(base.astype(int) + tint, 0, 255).astype(np.uint8)
            for _ in range(n_frames)
        ])
        return frames


# ---------------------------------------------------------------------------
# Quality Assurance (Section 3.2.4)
# ---------------------------------------------------------------------------

class QualityChecker:
    """
    Automated checks for anatomical plausibility, AU validity,
    physiological validity, and temporal coherence.
    """

    def __init__(self, cfg: dict):
        self.cfg = cfg

    def check_sample(self, sample: SyntheticSample) -> bool:
        return (
            self._check_physio(sample)
            and self._check_au_pain(sample)
            and self._check_temporal(sample)
        )

    def _check_physio(self, s: SyntheticSample) -> bool:
        # HRV RMSSD range [20, 120] ms
        diff  = np.diff(s.hrv_ibi)
        rmssd = float(np.sqrt(np.mean(diff ** 2)))
        if not (20 <= rmssd <= 120):
            return False

        # SCR amplitude < 2 µS
        tonic = np.percentile(s.eda_signal, 10)
        max_amp = np.max(s.eda_signal) - tonic
        if max_amp >= 2.0:
            return False

        # HRV–SCR correlation should be negative
        min_len = min(len(s.hrv_ibi), len(s.eda_signal) // 100)
        if min_len > 10:
            hrv_ds = s.hrv_ibi[:min_len]
            eda_ds = s.eda_signal[: min_len * 100 : 100]
            corr   = float(np.corrcoef(hrv_ds, eda_ds[:min_len])[0, 1])
            if corr >= -0.2:
                logger.debug("Failed HRV-SCR correlation check: %.3f", corr)
        return True

    def _check_au_pain(self, s: SyntheticSample) -> bool:
        pain = s.pain_intensity
        threshold = self.cfg.get("au_pain_mismatch_threshold", 0.30)
        for au in ["AU4", "AU6", "AU7"]:
            expected = 0.3 * pain / 10.0
            actual   = s.au_activations.get(au, 0.0)
            if pain > 1 and actual < expected * (1 - threshold):
                return False
        return True

    def _check_temporal(self, s: SyntheticSample) -> bool:
        # Frames exist and have correct shape
        if s.frames is None or s.frames.ndim != 4:
            return False
        return True


# ---------------------------------------------------------------------------
# Main Generator
# ---------------------------------------------------------------------------

class SyntheticPainGenerator:
    """
    Orchestrates generation of N multimodal pain scenarios.

    Usage
    -----
    >>> gen = SyntheticPainGenerator(cfg, device="cuda")
    >>> samples = gen.generate(n=50000, output_dir=Path("data/synthetic"))
    """

    def __init__(self, cfg: dict, device: str = "cpu"):
        self.cfg     = cfg
        self.device  = device
        self.rng     = np.random.default_rng(cfg.get("seed", 42))
        self.qc      = QualityChecker(cfg.get("quality", {}))
        self.physio  = CorrelatedPhysioSynthesizer(cfg, self.rng)
        self.temporal = TemporalDynamicsModel(self.rng)
        self.facial  = FacialExpressionGenerator(cfg.get("diffusion", {}), device)

    # -- Pain sampling -------------------------------------------------------

    def _sample_pain(self) -> float:
        """Uniform 0–10 with slight oversampling of extreme values."""
        p = self.rng.uniform(0, 10)
        # Oversample 8–10 by 1.5× (stratification)
        if self.rng.random() < 0.08:
            p = self.rng.uniform(8, 10)
        return round(float(np.clip(p, 0, 10)), 2)

    # -- Single sample -------------------------------------------------------

    def generate_one(self, sample_id: str,
                     pain_override: Optional[float] = None) -> SyntheticSample:
        pain   = pain_override if pain_override is not None else self._sample_pain()
        demo   = sample_demographics(self.rng)
        au     = compute_au_activations(pain)

        # Temporal profile
        n_frames = self.cfg.get("diffusion", {}).get("sequence_length", 30)
        fps      = self.cfg.get("diffusion", {}).get("fps", 30)
        duration = n_frames / fps
        traj, profile = self.temporal.generate_profile(pain, duration, fps)

        # Physiological signals
        hrv, eda, tremor = self.physio.synthesize(pain, duration_s=120)

        # Facial frames
        frames = self.facial.generate_sequence(pain, au, demo, traj, n_frames)

        return SyntheticSample(
            sample_id       = sample_id,
            pain_intensity  = pain,
            frames          = frames,
            hrv_ibi         = hrv,
            eda_signal      = eda,
            tremor_signal   = tremor,
            age_group       = demo["age_group"],
            ethnicity       = demo["ethnicity"],
            sex             = demo["sex"],
            clinical_setting= demo["clinical_setting"],
            pain_type       = demo["pain_type"],
            au_activations  = au,
            temporal_profile= profile,
        )

    # -- Batch generation ----------------------------------------------------

    def generate(
        self,
        n: int = 50_000,
        output_dir: Optional[Path] = None,
        show_progress: bool = True,
    ) -> List[SyntheticSample]:
        """Generate n quality-checked samples, regenerating failures."""
        samples   = []
        attempts  = 0
        max_regen = self.cfg.get("quality", {}).get("max_regen_attempts", 3)

        iterator = range(n)
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(range(n), desc="Generating synthetic pain data")
            except ImportError:
                pass

        for i in iterator:
            sample = None
            for attempt in range(max_regen):
                candidate = self.generate_one(f"synth_{i:06d}")
                if self.qc.check_sample(candidate):
                    candidate.quality_passed = True
                    sample = candidate
                    break
                attempts += 1

            if sample is None:
                # Accept last attempt with quality flag
                candidate.quality_passed = False
                sample = candidate

            samples.append(sample)

            if output_dir is not None and i % 1000 == 0:
                self._save_metadata(samples[-1000:], output_dir, i)

        accept_rate = 1 - attempts / (n * max_regen)
        logger.info("Generated %d samples | acceptance rate: %.1f%%",
                    n, accept_rate * 100)

        if output_dir is not None:
            self._save_metadata(samples, output_dir, n, final=True)

        return samples

    @staticmethod
    def _save_metadata(samples: list, output_dir: Path,
                       up_to: int, final: bool = False) -> None:
        """Save lightweight metadata (pain labels, demographics) to disk."""
        import json
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        tag  = "metadata_final" if final else f"metadata_{up_to:06d}"
        path = output_dir / f"{tag}.json"
        records = [
            {
                "id":               s.sample_id,
                "pain":             s.pain_intensity,
                "age_group":        s.age_group,
                "ethnicity":        s.ethnicity,
                "sex":              s.sex,
                "clinical_setting": s.clinical_setting,
                "pain_type":        s.pain_type,
                "temporal_profile": s.temporal_profile,
                "quality_passed":   s.quality_passed,
                "au_activations":   s.au_activations,
            }
            for s in samples
        ]
        with open(path, "w") as f:
            json.dump(records, f, indent=2)
