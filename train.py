"""
train.py  –  Main training script.

Run from the repo root:
    python train.py

In Colab after cloning the repo:
    !python train.py

Reads config from configs/efficientnet.yaml by default.
Override with --config configs/resnet50.yaml for the baseline.
"""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import yaml

from src.data import ChestXRayDataset, get_train_transforms, get_val_transforms
from src.models import PneumoniaDetectionAgent
from src.training import Trainer
from src.evaluation import evaluate_loader, plot_confusion_matrix, plot_training_curves
from src.evaluation.metrics import print_metrics
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Train the Pneumonia Detection Agent")
    p.add_argument("--config", default="configs/efficientnet.yaml")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Config: %s", args.config)

    # ------------------------------------------------------------------
    # 1. Datasets & dataloaders
    # ------------------------------------------------------------------
    data_root = Path(cfg["data"]["root"])
    train_ds  = ChestXRayDataset(data_root / "train", transform=get_train_transforms())
    val_ds    = ChestXRayDataset(data_root / "val",   transform=get_val_transforms())
    test_ds   = ChestXRayDataset(data_root / "test",  transform=get_val_transforms())

    logger.info("Train: %s | Val: %s | Test: %s",
                train_ds.class_counts(), val_ds.class_counts(), test_ds.class_counts())

    bs = cfg["training"]["batch_size"]
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,  num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)

    # ------------------------------------------------------------------
    # 2. Model (start frozen for Phase 1)
    # ------------------------------------------------------------------
    agent = PneumoniaDetectionAgent(
        backbone_name  = cfg["model"]["backbone"],
        dropout_p      = cfg["model"]["dropout"],
        frozen         = True,
        high_threshold = cfg["agent"]["high_threshold"],
        low_threshold  = cfg["agent"]["low_threshold"],
    )
    logger.info("Model: %s backbone", cfg["model"]["backbone"])

    # ------------------------------------------------------------------
    # 3. Train
    # ------------------------------------------------------------------
    trainer = Trainer(
        model          = agent,
        train_loader   = train_loader,
        val_loader     = val_loader,
        device         = device,
        pos_weight     = train_ds.pos_weight(),
        checkpoint_dir = cfg["paths"]["checkpoints"],
        phase1_epochs  = cfg["training"]["phase1_epochs"],
        phase1_lr      = cfg["training"]["phase1_lr"],
        phase2_epochs  = cfg["training"]["phase2_epochs"],
        phase2_lr      = cfg["training"]["phase2_lr"],
        patience       = cfg["training"]["patience"],
    )
    history = trainer.train()

    # ------------------------------------------------------------------
    # 4. Plot training curves
    # ------------------------------------------------------------------
    plot_training_curves(history, save_dir=cfg["paths"]["figures"])

    # ------------------------------------------------------------------
    # 5. Evaluate best model on test set
    # ------------------------------------------------------------------
    logger.info("Loading best checkpoint for test evaluation...")
    best_ckpt = Path(cfg["paths"]["checkpoints"]) / "best_model.pth"
    agent = load_checkpoint(agent, best_ckpt, device)

    metrics = evaluate_loader(agent, test_loader, device)
    print_metrics(metrics, split="Test")

    # Save confusion matrix
    plot_confusion_matrix(
        metrics["confusion_matrix"],
        save_path=str(Path(cfg["paths"]["figures"]) / "confusion_matrix.png"),
    )

    # Save metrics JSON for the report
    results_path = Path(cfg["paths"]["figures"]) / "test_metrics.json"
    serializable = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in metrics.items()}
    results_path.write_text(json.dumps(serializable, indent=2))
    logger.info("Metrics saved → %s", results_path)


if __name__ == "__main__":
    main()
