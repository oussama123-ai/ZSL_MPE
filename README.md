# Zero-Shot Multimodal Pain Estimation

> **Paper**: *Zero-Shot Multimodal Pain Estimation via Synthetic Pain Simulation and Domain-Invariant Learning*  
> **Authors**: Oussama El Othmani, Sami Naouali, Riadh Ouersighni

---

## Overview

A zero-shot multimodal pain estimation framework that trains **entirely on 50,000 synthetic scenarios** — no labeled pain data from real individuals required. The system combines:

- **Diffusion-based** facial expression synthesis (Stable Diffusion v2.1)
- **Physiologically-grounded** biosignal generation (HRV, EDA, tremor)
- **Transformer-based** multimodal fusion (Vision Transformer + 1D CNNs)
- **Three-stage domain alignment** (contrastive + adversarial + consistency)

### Key Results

| Dataset | Metric | Score | Gap vs. Supervised |
|---|---|---|---|
| UNBC-McMaster | MAE | 0.89 | 31% |
| BioVid | Accuracy | 78.3% | 10.5% |
| Neonatal | Accuracy | 81.5% | Exceeds clinical by +12.7% |
| Cross-dataset avg | Generalization | +40.4% | vs. supervised transfer |

---

## Project Structure

```
zero_shot_pain_estimation/
├── configs/
│   └── default.yaml              # All hyperparameters
├── data/
│   ├── synthetic_generator.py    # Diffusion + physiological signal synthesis
│   ├── dataset.py                # PyTorch Dataset classes
│   └── augmentations.py          # Strong augmentation pipeline
├── models/
│   ├── visual_encoder.py         # ViT-Base/16 + temporal transformer
│   ├── physio_encoder.py         # 1D CNN encoders for HRV/EDA/tremor
│   ├── context_encoder.py        # Demographic & clinical embeddings
│   ├── fusion_transformer.py     # Cross-modal attention fusion
│   ├── domain_alignment.py       # Discriminator + GRL
│   └── pain_estimator.py         # Full model wrapper
├── training/
│   ├── losses.py                 # MSE, contrastive, adversarial, consistency
│   ├── trainer.py                # Three-stage training loop
│   └── scheduler.py              # LR warmup + cosine annealing
├── evaluation/
│   ├── metrics.py                # MAE, RMSE, PCC, ICC, AUC, fairness
│   └── evaluator.py              # Evaluation pipeline
├── utils/
│   ├── logger.py                 # Logging utilities
│   └── checkpoint.py             # Save/load checkpoints
├── scripts/
│   ├── generate_synthetic.py     # Generate 50K synthetic scenarios
│   ├── train.py                  # Main training entry point
│   └── evaluate.py               # Evaluation entry point
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/<your-username>/zero-shot-pain-estimation.git
cd zero-shot-pain-estimation
pip install -r requirements.txt
```

---

## Quick Start

### 1. Generate Synthetic Data

```bash
python scripts/generate_synthetic.py \
    --n_samples 50000 \
    --output_dir data/synthetic \
    --config configs/default.yaml
```

### 2. Train the Model

```bash
python scripts/train.py \
    --config configs/default.yaml \
    --synthetic_dir data/synthetic \
    --real_unlabeled_dir data/real_unlabeled \
    --output_dir checkpoints/
```

### 3. Evaluate

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/best_model.pt \
    --dataset unbc \
    --data_dir data/UNBC-McMaster \
    --config configs/default.yaml
```

---

## Three-Stage Training Protocol

| Stage | Epochs | Objective |
|---|---|---|
| 1 — Synthetic Pre-training | 1–100 | Supervised pain learning on synthetic data |
| 2 — Domain Alignment | 101–150 | Contrastive + adversarial + consistency losses |
| 3 — Consistency Refinement | 151–170 | Stability on unlabeled real data |

---

## Citation

```bibtex
@article{elothmani2025zeroshotpain,
  title={Zero-Shot Multimodal Pain Estimation via Synthetic Pain Simulation and Domain-Invariant Learning},
  author={El Othmani, Oussama and Naouali, Sami and Ouersighni, Riadh},
  year={2025}
}
```

---

## License

This repository is released for academic research purposes only.
