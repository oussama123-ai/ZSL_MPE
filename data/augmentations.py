"""
Augmentation Pipeline
=====================
Strong and weak augmentations for video frames and physiological signals,
as described in Section 3.4.3 of the paper.
"""

from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class VideoAugmentor:
    """
    Applies frame-level augmentations to (T, H, W, 3) uint8 arrays.

    Weak mode:  ColorJitter ±0.1, RandomCrop, HorizontalFlip 50%
    Strong mode: ColorJitter ±0.3, RandomGrayscale 10%, GaussianBlur,
                 RandomErasing 20%, temporal shift, frame dropout 10%
    """

    def __init__(self, cfg: dict, strong: bool = True):
        a = cfg.get("augmentation", {})
        if strong:
            brightness  = a.get("brightness",  0.3)
            contrast    = a.get("contrast",     0.3)
            saturation  = a.get("saturation",   0.2)
            grayscale   = a.get("grayscale_prob", 0.10)
            blur_lo     = a.get("blur_sigma_min",  0.1)
            blur_hi     = a.get("blur_sigma_max",  2.0)
            erase_prob  = a.get("erasing_prob",    0.20)
        else:
            brightness  = 0.1
            contrast    = 0.1
            saturation  = 0.0
            grayscale   = 0.0
            blur_lo     = 0.0
            blur_hi     = 0.0
            erase_prob  = 0.0

        self.strong       = strong
        self.color_jitter = T.ColorJitter(brightness, contrast, saturation)
        self.grayscale_p  = grayscale
        self.blur_lo      = blur_lo
        self.blur_hi      = blur_hi
        self.erase_prob   = erase_prob
        self.h_flip_p     = 0.5
        self.frame_drop_p = a.get("frame_dropout_prob", 0.10) if strong else 0.0

    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """frames: (T, H, W, 3) uint8 → (T, H, W, 3) uint8."""
        T_frames, H, W, C = frames.shape
        out = []

        # Decide augmentation parameters once per sequence
        do_hflip  = random.random() < self.h_flip_p
        do_gray   = random.random() < self.grayscale_p
        do_blur   = self.blur_hi > 0 and random.random() < 0.5
        blur_sig  = random.uniform(self.blur_lo, self.blur_hi)
        do_erase  = random.random() < self.erase_prob

        for i in range(T_frames):
            img = frames[i]  # (H, W, 3)

            # Frame dropout — replace with adjacent frame
            if self.frame_drop_p > 0 and random.random() < self.frame_drop_p:
                prev = max(i - 1, 0)
                img = frames[prev]

            # To PIL for torchvision transforms
            from PIL import Image
            pil = Image.fromarray(img)
            pil = self.color_jitter(pil)

            if do_gray:
                pil = TF.rgb_to_grayscale(pil, num_output_channels=3)
            if do_blur:
                pil = TF.gaussian_blur(pil, kernel_size=5, sigma=blur_sig)
            if do_hflip:
                pil = TF.hflip(pil)

            img = np.array(pil, dtype=np.uint8)

            if do_erase:
                img = self._random_erase(img, H, W)

            out.append(img)

        return np.stack(out)

    @staticmethod
    def _random_erase(img: np.ndarray, H: int, W: int) -> np.ndarray:
        """Zero out a random rectangular patch."""
        h = random.randint(H // 8, H // 3)
        w = random.randint(W // 8, W // 3)
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)
        img = img.copy()
        img[y:y + h, x:x + w] = 0
        return img


class PhysioAugmentor:
    """
    Augments 1-D physiological signals (HRV IBI, EDA, tremor).

    Operations (Section 3.4.3):
      - Additive Gaussian noise (SNR 15–25 dB)
      - Time warping
      - Magnitude warping
    """

    def __init__(self, cfg: dict):
        a            = cfg.get("augmentation", {})
        self.snr_min = a.get("physio_noise_snr_min", 15)
        self.snr_max = a.get("physio_noise_snr_max", 25)

    def __call__(
        self,
        hrv: np.ndarray,
        eda: np.ndarray,
        tremor: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        hrv    = self._augment_signal(hrv)
        eda    = self._augment_signal(eda)
        tremor = self._augment_signal(tremor)
        return hrv, eda, tremor

    def _augment_signal(self, sig: np.ndarray) -> np.ndarray:
        sig = self._add_noise(sig)
        if random.random() < 0.3:
            sig = self._magnitude_warp(sig)
        if random.random() < 0.3:
            sig = self._time_warp(sig)
        return sig

    def _add_noise(self, sig: np.ndarray) -> np.ndarray:
        snr_db  = random.uniform(self.snr_min, self.snr_max)
        power   = np.mean(sig ** 2) + 1e-8
        noise_p = power / (10 ** (snr_db / 10))
        noise   = np.random.normal(0, np.sqrt(noise_p), len(sig))
        return (sig + noise).astype(np.float32)

    @staticmethod
    def _magnitude_warp(sig: np.ndarray, sigma: float = 0.2) -> np.ndarray:
        """Smooth multiplicative magnitude perturbation."""
        n      = len(sig)
        knots  = np.random.normal(1.0, sigma, 4)
        warp   = np.interp(np.linspace(0, 1, n),
                           np.linspace(0, 1, 4), knots)
        return (sig * warp).astype(np.float32)

    @staticmethod
    def _time_warp(sig: np.ndarray) -> np.ndarray:
        """Smooth time-axis warping."""
        n       = len(sig)
        anchors = np.sort(np.random.uniform(0.1, 0.9, 3))
        anchors = np.concatenate([[0.0], anchors, [1.0]])
        warped  = anchors + np.random.normal(0, 0.02, len(anchors))
        warped  = np.clip(np.sort(warped), 0, 1)
        new_t   = np.interp(np.linspace(0, 1, n), anchors, warped)
        new_t   = np.clip(new_t, 0, 1) * (n - 1)
        return np.interp(new_t, np.arange(n), sig).astype(np.float32)
