"""
colab_setup.py
Run this at the top of your Colab session BEFORE anything else.
It handles: Kaggle auth, dataset download, repo clone, dependency install.

Usage in Colab:
    !python scripts/colab_setup.py
"""

import os
import subprocess
import sys

def run(cmd):
    print(f"$ {cmd}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR: command failed with exit code {result.returncode}")
        sys.exit(1)

print("=" * 60)
print("Pneumonia Detection Agent — Colab Setup")
print("=" * 60)

# 1. Install dependencies
print("\n[1/4] Installing dependencies...")
run("pip install -q torch torchvision scikit-learn matplotlib pyyaml Pillow")

# 2. Mount Google Drive (for saving checkpoints across sessions)
print("\n[2/4] Mounting Google Drive...")
try:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=True)
    print("Drive mounted at /content/drive")
except ImportError:
    print("Not running in Colab — skipping Drive mount.")

# 3. Download dataset via Kaggle API
print("\n[3/4] Downloading Kaggle dataset...")
print("  → You need a kaggle.json API token.")
print("  → Get it from: https://www.kaggle.com/settings → API → Create New Token")
print("  → Upload kaggle.json when prompted.\n")

try:
    from google.colab import files
    print("Upload your kaggle.json now:")
    uploaded = files.upload()
    os.makedirs(os.path.expanduser("~/.config/kaggle"), exist_ok=True)
    os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
    with open(os.path.expanduser("~/.kaggle/kaggle.json"), "wb") as f:
        f.write(uploaded["kaggle.json"])
    os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)
    run("pip install -q kaggle")
    run("kaggle datasets download -d paultimothymooney/chest-xray-pneumonia")
    run("unzip -q chest-xray-pneumonia.zip -d data/")
    # The Kaggle zip unpacks to chest_xray/ — rename to data/raw/
    if os.path.exists("data/chest_xray"):
        run("mv data/chest_xray data/raw")
    print("Dataset ready at data/raw/")
except ImportError:
    print("Not in Colab. Download manually from:")
    print("https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia")
    print("Then place it at data/raw/ (with train/ val/ test/ subdirectories)")

# 4. Verify directory structure
print("\n[4/4] Verifying dataset structure...")
expected = ["data/raw/train/NORMAL", "data/raw/train/PNEUMONIA",
            "data/raw/val/NORMAL",   "data/raw/val/PNEUMONIA",
            "data/raw/test/NORMAL",  "data/raw/test/PNEUMONIA"]
all_ok = True
for d in expected:
    exists = os.path.isdir(d)
    status = "OK" if exists else "MISSING"
    print(f"  [{status}] {d}")
    if not exists:
        all_ok = False

if all_ok:
    print("\nSetup complete! Run training with:")
    print("  !python train.py")
else:
    print("\nSome directories are missing — check the dataset download.")
