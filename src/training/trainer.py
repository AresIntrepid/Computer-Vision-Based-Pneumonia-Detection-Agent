import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from src.training.losses import get_weighted_bce_loss
from src.utils.checkpoint import save_checkpoint
from src.utils.logger import get_logger

logger = get_logger(__name__)


class Trainer:
    def __init__(self, model, train_loader, val_loader, device, pos_weight,
                 checkpoint_dir="results/checkpoints", phase1_epochs=5,
                 phase1_lr=1e-3, phase2_epochs=20, phase2_lr=1e-4, patience=5):
        self.model         = model.to(device)
        self.train_loader  = train_loader
        self.val_loader    = val_loader
        self.device        = device
        self.ckpt_dir      = Path(checkpoint_dir)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self.phase1_epochs = phase1_epochs
        self.phase1_lr     = phase1_lr
        self.phase2_epochs = phase2_epochs
        self.phase2_lr     = phase2_lr
        self.patience      = patience
        self.criterion     = get_weighted_bce_loss(pos_weight, device)
        self.history       = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    def train(self) -> dict:
        logger.info("PHASE 1 — Feature extraction (backbone frozen)")
        self._run_phase(self.phase1_epochs, self.phase1_lr, "phase1_best.pth", early_stop=False)
        logger.info("PHASE 2 — Full fine-tuning")
        self.model.unfreeze()
        self._run_phase(self.phase2_epochs, self.phase2_lr, "best_model.pth", early_stop=True)
        logger.info("Training complete. Best model → %s/best_model.pth", self.ckpt_dir)
        logger.info("History lengths: %s", {k: len(v) for k, v in self.history.items()})
        # Return a copy so callers always get the populated dict
        return {k: list(v) for k, v in self.history.items()}

    def _run_phase(self, n_epochs, lr, save_name, early_stop):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()), lr=lr)
        best_val_loss, no_improve = float("inf"), 0
        for epoch in range(1, n_epochs + 1):
            t0 = time.time()
            tr_loss, tr_acc = self._train_epoch(optimizer)
            vl_loss, vl_acc = self._val_epoch()
            elapsed = time.time() - t0

            self.history["train_loss"].append(tr_loss)
            self.history["val_loss"].append(vl_loss)
            self.history["train_acc"].append(tr_acc)
            self.history["val_acc"].append(vl_acc)

            logger.info("Epoch %02d/%02d | train loss=%.4f acc=%.3f | val loss=%.4f acc=%.3f | %.1fs",
                        epoch, n_epochs, tr_loss, tr_acc, vl_loss, vl_acc, elapsed)

            if vl_loss < best_val_loss:
                best_val_loss, no_improve = vl_loss, 0
                save_checkpoint(self.model, self.ckpt_dir / save_name)
                logger.info("  ✓ New best saved.")
            else:
                no_improve += 1

            if early_stop and no_improve >= self.patience:
                logger.info("Early stopping at epoch %d.", epoch)
                break

    def _train_epoch(self, optimizer):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.float().to(self.device)
            optimizer.zero_grad()
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * images.size(0)
            correct    += ((torch.sigmoid(logits) >= 0.5).long() == labels.long()).sum().item()
            total      += images.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def _val_epoch(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in self.val_loader:
            images, labels = images.to(self.device), labels.float().to(self.device)
            logits = self.model(images)
            loss   = self.criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct    += ((torch.sigmoid(logits) >= 0.5).long() == labels.long()).sum().item()
            total      += images.size(0)
        return total_loss / total, correct / total