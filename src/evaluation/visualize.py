from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm_module
import torch
import torch.nn.functional as F
from PIL import Image


def plot_confusion_matrix(cm, class_names=["NORMAL", "PNEUMONIA"],
                          save_path="results/figures/confusion_matrix.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="Predicted", ylabel="True", title="Confusion Matrix")
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix saved → {save_path}")


def plot_training_curves(history, save_dir="results/figures"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    for (k1, k2), title, fname in [
        (("train_loss", "val_loss"), "Loss", "loss_curves.png"),
        (("train_acc",  "val_acc"),  "Accuracy", "accuracy_curves.png"),
    ]:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs, history[k1], label="Train")
        ax.plot(epochs, history[k2], label="Val")
        ax.set(xlabel="Epoch", ylabel=title, title=f"Training & Validation {title}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(save_dir / fname, dpi=150)
        plt.close(fig)
    print(f"Curves saved → {save_dir}/")


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self._gradients = self._activations = None
        target_layer.register_forward_hook(lambda _, __, o: setattr(self, "_activations", o.detach()))
        target_layer.register_full_backward_hook(lambda _, __, g: setattr(self, "_gradients", g[0].detach()))

    def __call__(self, tensor):
        self.model.eval()
        tensor = tensor.clone().requires_grad_(True)
        self.model(tensor).backward()
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._activations).sum(dim=1).squeeze()).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

    @staticmethod
    def overlay(pil_image, heatmap, alpha=0.5):
        h = Image.fromarray(np.uint8(255 * heatmap)).resize(pil_image.size, resample=Image.BILINEAR)
        colored = np.uint8(255 * cm_module.get_cmap("jet")(np.array(h) / 255.0)[:, :, :3])
        return Image.blend(pil_image.convert("RGB"), Image.fromarray(colored), alpha=alpha)


def save_gradcam(model, image_path, val_transforms, save_path, device):
    gcam = GradCAM(model, model.backbone.features[-1])
    pil_img = Image.open(image_path).convert("RGB")
    tensor  = val_transforms(pil_img).unsqueeze(0).to(device)
    overlay = GradCAM.overlay(pil_img, gcam(tensor))
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    overlay.save(save_path)
    print(f"Grad-CAM saved → {save_path}")