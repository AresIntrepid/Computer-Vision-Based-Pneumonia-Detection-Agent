"""
run_agent.py  –  Run the trained agent on one or more X-ray images.

This is what makes it an AGENT, not just a classifier.
For each image it:
  1. Perceives  – loads and preprocesses the X-ray
  2. Decides    – runs EfficientNet-B3 to get a probability score
  3. Acts       – applies the decision policy → risk tier + recommendation
  4. Outputs    – prints a structured result and optionally saves a Grad-CAM overlay

Usage:
    # Single image
    python run_agent.py --image path/to/xray.jpeg

    # Folder of images
    python run_agent.py --image path/to/folder/

    # With Grad-CAM visualisation
    python run_agent.py --image path/to/xray.jpeg --gradcam

    # Custom checkpoint
    python run_agent.py --image path/to/xray.jpeg --checkpoint results/checkpoints/best_model.pth
"""

import argparse
from pathlib import Path

import torch
import yaml

from src.data.transforms import get_test_transforms
from src.models import PneumoniaDetectionAgent
from src.utils.checkpoint import load_checkpoint
from src.utils.logger import get_logger
from src.evaluation.visualize import save_gradcam

logger = get_logger(__name__)

SUPPORTED = {".jpeg", ".jpg", ".png"}


def parse_args():
    p = argparse.ArgumentParser(description="Pneumonia Detection Agent — Inference")
    p.add_argument("--image",      required=True, help="Path to an X-ray image or a folder of images")
    p.add_argument("--checkpoint", default="results/checkpoints/best_model.pth")
    p.add_argument("--config",     default="configs/efficientnet.yaml")
    p.add_argument("--gradcam",    action="store_true", help="Save a Grad-CAM overlay for each image")
    p.add_argument("--gradcam_dir", default="results/figures/gradcam")
    return p.parse_args()


def collect_images(path: str) -> list[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    if p.is_dir():
        return sorted(f for f in p.iterdir() if f.suffix.lower() in SUPPORTED)
    raise FileNotFoundError(f"No file or directory found at: {path}")


def main():
    args      = parse_args()
    cfg       = yaml.safe_load(open(args.config))
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = get_test_transforms()

    # ------------------------------------------------------------------
    # Load trained agent
    # ------------------------------------------------------------------
    agent = PneumoniaDetectionAgent(
        backbone_name  = cfg["model"]["backbone"],
        dropout_p      = cfg["model"]["dropout"],
        frozen         = False,
        high_threshold = cfg["agent"]["high_threshold"],
        low_threshold  = cfg["agent"]["low_threshold"],
    )
    agent = load_checkpoint(agent, args.checkpoint, device)
    agent.eval()
    logger.info("Agent loaded from %s", args.checkpoint)

    # ------------------------------------------------------------------
    # Run on each image
    # ------------------------------------------------------------------
    images = collect_images(args.image)
    logger.info("Running agent on %d image(s)...\n", len(images))

    summary = {"HIGH": 0, "BORDERLINE": 0, "NORMAL": 0}

    for img_path in images:
        from PIL import Image
        pil_img = Image.open(img_path).convert("RGB")
        tensor  = transform(pil_img).unsqueeze(0).to(device)

        result = agent.run(tensor, image_path=str(img_path))
        print(result)

        summary[result.risk_tier] += 1

        # Optional Grad-CAM
        if args.gradcam:
            gradcam_path = Path(args.gradcam_dir) / f"gradcam_{img_path.stem}.png"
            save_gradcam(agent, str(img_path), transform, str(gradcam_path), device)

    # ------------------------------------------------------------------
    # Batch summary
    # ------------------------------------------------------------------
    if len(images) > 1:
        total = len(images)
        print("\n" + "=" * 52)
        print("  Batch Summary")
        print("=" * 52)
        print(f"  Total images   : {total}")
        print(f"  HIGH risk      : {summary['HIGH']}  ({summary['HIGH']/total:.0%})")
        print(f"  BORDERLINE     : {summary['BORDERLINE']}  ({summary['BORDERLINE']/total:.0%})")
        print(f"  NORMAL         : {summary['NORMAL']}  ({summary['NORMAL']/total:.0%})")
        print("=" * 52)


if __name__ == "__main__":
    main()
