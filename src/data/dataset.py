from pathlib import Path
from typing import Callable, Optional, Tuple
from PIL import Image
from torch.utils.data import Dataset

LABEL_MAP = {"NORMAL": 0, "PNEUMONIA": 1}


class ChestXRayDataset(Dataset):
    def __init__(self, root_dir, transform=None, return_path=False):
        self.root_dir    = Path(root_dir)
        self.transform   = transform
        self.return_path = return_path
        self.samples     = []
        self._load_samples()

    def _load_samples(self):
        for class_name, label in LABEL_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                raise FileNotFoundError(f"Missing: {class_dir}")
            for f in sorted(class_dir.iterdir()):
                if f.suffix.lower() in {".jpeg", ".jpg", ".png"}:
                    self.samples.append((f, label))
        if not self.samples:
            raise RuntimeError(f"No images found under {self.root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        if self.return_path:
            return image, label, str(img_path)
        return image, label

    def class_counts(self):
        inv = {v: k for k, v in LABEL_MAP.items()}
        counts = {k: 0 for k in LABEL_MAP}
        for _, label in self.samples:
            counts[inv[label]] += 1
        return counts

    def pos_weight(self):
        c = self.class_counts()
        return c["NORMAL"] / c["PNEUMONIA"]