# Chest X-Ray Pneumonia Detection

A computer vision agent that classifies chest X-ray images as **NORMAL** or **PNEUMONIA** using a fine-tuned pre-trained CNN.

---

## Overview
This project fine-tunes a pre-trained CNN (ResNet-50 or EfficientNet) on a labeled dataset of ~5,800 chest X-rays. The goal is a decision-support tool that can flag likely pneumonia cases, reducing the burden on radiologists screening large volumes of X-rays.

---

## Dataset
**Chest X-Ray Images (Pneumonia)** – Kaggle  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

- ~5,800 labeled JPEG images
- Classes: `NORMAL`, `PNEUMONIA`
- Pre-split into `train/`, `val/`, and `test/` directories

> Dataset is not committed to this repo. Download it from Kaggle and place it in `data/raw/`.

---

## Repository Structure
```
Computer-Vision-Based-Pneumonia-Detection-Agent/
├── data/
│   ├── raw/                    # Original downloaded dataset, never modified
│   ├── processed/              # Cleaned, resized, normalized data
│   └── splits/                 # Train / val / test split manifests (CSV or JSON)
├── src/
│   ├── data/
│   │   ├── dataset.py          # PyTorch Dataset class
│   │   ├── transforms.py       # Augmentation and preprocessing pipelines
│   │   └── split.py            # Script to generate train/val/test splits
│   ├── models/
│   │   ├── backbone.py         # Pre-trained backbone loader (ResNet-50, EfficientNet)
│   │   ├── head.py             # Binary classification head
│   │   └── agent.py            # Full model pipeline (backbone + head + inference logic)
│   ├── training/
│   │   ├── trainer.py          # Training loop
│   │   ├── losses.py           # Weighted BCE, focal loss
│   │   └── scheduler.py        # LR schedulers
│   ├── evaluation/
│   │   ├── metrics.py          # Accuracy, F1, AUC-ROC, precision, recall
│   │   ├── visualize.py        # Confusion matrix, Grad-CAM overlays
│   │   └── benchmark.py        # Full eval on test set, export results
│   └── utils/
│       ├── config.py           # Config loader
│       ├── logger.py           # Logging setup
│       └── checkpoint.py       # Save / load model checkpoints
├── configs/
│   ├── base.yaml               # Base hyperparameters and paths
│   ├── resnet50.yaml           # ResNet-50 specific config
│   └── efficientnet.yaml       # EfficientNet specific config
├── scripts/
│   ├── train.sh                # Launch training
│   ├── evaluate.sh             # Run evaluation
│   └── download_data.sh        # Automate Kaggle dataset download
├── results/
│   ├── checkpoints/            # Saved model weights (.pth files)
│   ├── logs/                   # Training logs (loss, metrics per epoch)
│   └── figures/                # Plots, confusion matrices, sample predictions
├── report/
│   ├── final_report.pdf
│   └── presentation.pptx
├── tests/
│   ├── test_dataset.py         # Unit tests for data loading
│   ├── test_model.py           # Unit tests for model forward pass
│   └── test_metrics.py         # Unit tests for metric calculations
├── .gitignore
├── requirements.txt
├── environment.yml
└── README.md
```

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/AresIntrepid/189-Project.git
cd Computer-Vision-Based-Pneumonia-Detection-Agent
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract into `data/raw/`.

### 4. Train the model
```bash
bash scripts/train.sh
```

### 5. Evaluate
```bash
bash scripts/evaluate.sh
```

---

## Approach
- **Base model:** ResNet-50 or EfficientNet (pre-trained on ImageNet)
- **Fine-tuning:** Replace final classification layer; train on chest X-ray data
- **Preprocessing:** Resize to 224×224, normalize, data augmentation (flips, rotation)
- **Loss:** Binary cross-entropy with class weighting to handle imbalance
- **Metrics:** Accuracy, Precision, Recall, F1-Score, AUC-ROC

---

## Status
- [x] Dataset identified and downloaded
- [x] Initial data exploration complete
- [ ] Preprocessing pipeline
- [ ] Model training
- [ ] Evaluation
- [ ] Final report

---

## References
- Kaggle Dataset: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- He et al. (2016). Deep Residual Learning for Image Recognition.
- Tan & Le (2019). EfficientNet: Rethinking Model Scaling for CNNs.
