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

> Dataset is not committed to this repo. Download it from Kaggle and place it in `data/`.

---

## Repository Structure
```
189-Project/
├── data/                  # Dataset (not committed, download from Kaggle)
│   ├── train/
│   ├── val/
│   └── test/
├── notebooks/             # Jupyter notebooks for EDA and experiments
├── src/                   # Source code
│   ├── preprocess.py      # Data loading and preprocessing
│   ├── model.py           # Model architecture and fine-tuning
│   ├── train.py           # Training loop
│   └── evaluate.py        # Evaluation and metrics
├── results/               # Saved model weights, plots, metrics
├── report/                # Final report and presentation
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Setup & Usage

### 1. Clone the repo
```bash
git clone https://github.com/AresIntrepid/189-Project.git
cd 189-Project
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download from [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) and extract into the `data/` folder.

### 4. Train the model
```bash
python src/train.py
```

### 5. Evaluate
```bash
python src/evaluate.py
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
