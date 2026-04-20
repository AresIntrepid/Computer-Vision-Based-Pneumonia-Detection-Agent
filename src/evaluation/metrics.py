import numpy as np
import torch
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)


def compute_metrics(labels, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy":         accuracy_score(labels, preds),
        "precision":        precision_score(labels, preds, zero_division=0),
        "recall":           recall_score(labels, preds, zero_division=0),
        "f1":               f1_score(labels, preds, zero_division=0),
        "auc_roc":          roc_auc_score(labels, probs),
        "confusion_matrix": confusion_matrix(labels, preds),
    }


@torch.no_grad()
def evaluate_loader(model, loader, device):
    model.eval()
    all_labels, all_probs = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        all_probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
    return compute_metrics(np.array(all_labels), np.array(all_probs))


def print_metrics(metrics, split="Test"):
    cm = metrics["confusion_matrix"]
    print(f"\n{'='*52}")
    print(f"  {split} Set Results")
    print(f"{'='*52}")
    print(f"  Accuracy  : {metrics['accuracy']*100:.1f}%")
    print(f"  Precision : {metrics['precision']:.4f}")
    print(f"  Recall    : {metrics['recall']:.4f}")
    print(f"  F1-Score  : {metrics['f1']:.4f}")
    print(f"  AUC-ROC   : {metrics['auc_roc']:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"                    Pred: NORMAL  Pred: PNEUMONIA")
    print(f"  Actual: NORMAL       {cm[0][0]:>5}           {cm[0][1]:>5}")
    print(f"  Actual: PNEUMONIA    {cm[1][0]:>5}           {cm[1][1]:>5}")
    print(f"{'='*52}\n")