"""
agent.py  –  PneumoniaDetectionAgent

This is the full agent pipeline, not just a classifier.

Architecture:
    Perceive  →  Decide (EfficientNet-B3)  →  Act (decision policy)  →  Output

The "Act" layer applies a tiered risk policy to the model's probability score
and returns a structured AgentResult with a risk tier, recommendation, and
confidence — mimicking how a real clinical decision-support tool would behave.

Risk tiers (tunable in configs/efficientnet.yaml):
    HIGH   : prob >= 0.75  →  urgent review flagged
    BORDERLINE : 0.50–0.75 →  standard physician review
    NORMAL : prob <  0.50  →  likely normal, log and pass
"""

from __future__ import annotations

import dataclasses
from typing import Optional

import torch
import torch.nn as nn
from PIL import Image

from .backbone import get_backbone, unfreeze_backbone
from .head import BinaryClassificationHead


# ---------------------------------------------------------------------------
# Structured output
# ---------------------------------------------------------------------------

@dataclasses.dataclass
class AgentResult:
    """Everything the agent returns for a single X-ray."""
    probability:    float          # P(PNEUMONIA), 0–1
    prediction:     str            # "PNEUMONIA" or "NORMAL"
    risk_tier:      str            # "HIGH" | "BORDERLINE" | "NORMAL"
    confidence:     float          # distance from 0.5, rescaled to 0–1
    recommendation: str            # plain-English action string
    image_path:     Optional[str]  # source image if provided

    def __str__(self) -> str:
        lines = [
            "─" * 52,
            f"  Pneumonia Detection Agent — Result",
            "─" * 52,
            f"  Image      : {self.image_path or 'tensor input'}",
            f"  Probability: {self.probability:.1%}  →  {self.prediction}",
            f"  Risk tier  : {self.risk_tier}",
            f"  Confidence : {self.confidence:.1%}",
            f"  Action     : {self.recommendation}",
            "─" * 52,
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Decision policy
# ---------------------------------------------------------------------------

class DecisionPolicy:
    """
    Maps a probability score to a risk tier and recommendation.

    Thresholds are conservative by design: in a screening context,
    false negatives (missed pneumonia) are more dangerous than false positives.
    """

    def __init__(
        self,
        high_threshold: float = 0.75,
        low_threshold:  float = 0.50,
    ) -> None:
        self.high_threshold = high_threshold
        self.low_threshold  = low_threshold

    def evaluate(self, prob: float) -> tuple[str, str, str]:
        """
        Returns (prediction, risk_tier, recommendation).
        """
        if prob >= self.high_threshold:
            return (
                "PNEUMONIA",
                "HIGH",
                "Flag for URGENT physician review. Strong radiological indicators "
                "of pneumonia detected. Do not use as sole diagnostic.",
            )
        elif prob >= self.low_threshold:
            return (
                "PNEUMONIA",
                "BORDERLINE",
                "Flag for standard physician review. Model is uncertain — "
                "borderline presentation. Clinical correlation required.",
            )
        else:
            return (
                "NORMAL",
                "NORMAL",
                "No significant pneumonia indicators detected. "
                "Recommend routine follow-up per clinical protocol.",
            )

    @staticmethod
    def confidence(prob: float) -> float:
        """How far from the decision boundary (0.5), scaled to [0, 1]."""
        return abs(prob - 0.5) * 2.0


# ---------------------------------------------------------------------------
# Main agent
# ---------------------------------------------------------------------------

class PneumoniaDetectionAgent(nn.Module):
    """
    Full agent: backbone + classification head + decision policy.

    Usage (training):
        agent = PneumoniaDetectionAgent(frozen=True)   # Phase 1
        # ... train head only ...
        agent.unfreeze()                               # Phase 2
        # ... full fine-tuning ...

    Usage (inference):
        result = agent.run(image_tensor)   # returns AgentResult
        print(result)
    """

    def __init__(
        self,
        backbone_name:  str   = "efficientnet_b3",
        dropout_p:      float = 0.3,
        frozen:         bool  = True,
        high_threshold: float = 0.75,
        low_threshold:  float = 0.50,
    ) -> None:
        super().__init__()
        self.backbone, out_features = get_backbone(backbone_name, frozen=frozen)
        self.head   = BinaryClassificationHead(out_features, dropout_p)
        self.policy = DecisionPolicy(high_threshold, low_threshold)

    # ------------------------------------------------------------------
    # nn.Module forward — returns raw logits for training
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns raw logits (scalar per image). Used by the training loop."""
        features = self.backbone(x)
        return self.head(features)

    # ------------------------------------------------------------------
    # Agent interface — returns structured AgentResult for inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def run(
        self,
        image_tensor: torch.Tensor,
        image_path:   Optional[str] = None,
    ) -> AgentResult:
        """
        Full agent pipeline for a single image.

        Args:
            image_tensor: preprocessed tensor, shape (1, 3, 224, 224)
            image_path:   optional source path for display in the result

        Returns:
            AgentResult with probability, prediction, risk tier, recommendation.
        """
        self.eval()
        logit = self(image_tensor)
        prob  = torch.sigmoid(logit).item()

        prediction, risk_tier, recommendation = self.policy.evaluate(prob)
        confidence = self.policy.confidence(prob)

        return AgentResult(
            probability    = prob,
            prediction     = prediction,
            risk_tier      = risk_tier,
            confidence     = confidence,
            recommendation = recommendation,
            image_path     = image_path,
        )

    @torch.no_grad()
    def run_batch(
        self,
        image_tensors: torch.Tensor,
        image_paths:   Optional[list[str]] = None,
    ) -> list[AgentResult]:
        """Run the agent on a batch. Returns one AgentResult per image."""
        self.eval()
        logits = self(image_tensors)
        probs  = torch.sigmoid(logits).cpu().tolist()
        paths  = image_paths or [None] * len(probs)

        results = []
        for prob, path in zip(probs, paths):
            pred, tier, rec = self.policy.evaluate(prob)
            results.append(AgentResult(
                probability    = prob,
                prediction     = pred,
                risk_tier      = tier,
                confidence     = self.policy.confidence(prob),
                recommendation = rec,
                image_path     = path,
            ))
        return results

    # ------------------------------------------------------------------
    # Training helpers
    # ------------------------------------------------------------------

    def unfreeze(self) -> None:
        """Switch from Phase 1 (frozen backbone) to Phase 2 (full fine-tuning)."""
        unfreeze_backbone(self.backbone)
