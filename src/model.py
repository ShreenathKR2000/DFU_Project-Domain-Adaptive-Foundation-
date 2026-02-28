"""
DINOv2 ViT-B/14 backbone with LoRA adapters and a 4-class head.

Trainable parameters: LoRA adapters (Q/V projections) + classification head.
Everything else in the backbone stays frozen.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import Dinov2Model


LABEL_NAMES = ["none", "infection", "ischaemia", "both"]
NUM_CLASSES = len(LABEL_NAMES)


class DFUDinoLoRA(nn.Module):
    """Frozen DINOv2-ViT-B/14 + LoRA adapters on Q/V projections + linear head.

    Parameters
    ----------
    model_name : str
        Hugging Face model id (default ``facebook/dinov2-vitb14``).
    num_classes : int
        Number of output classes (default 4).
    lora_rank : int
        LoRA low-rank dimension (default 16).
    lora_alpha : int
        LoRA scaling factor (default 32).
    lora_dropout : float
        Dropout inside LoRA layers (default 0.1).
    """

    def __init__(
        self,
        model_name = "facebook/dinov2-base",
        num_classes: int = NUM_CLASSES,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ) -> None:
        super().__init__()

        # 1. Load pretrained backbone and freeze it.
        backbone = Dinov2Model.from_pretrained(model_name)
        for param in backbone.parameters():
            param.requires_grad = False

        # 2. Inject LoRA into Query and Value projection layers.
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=["query", "value"],
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_config)

        # 3. Classification head on top of the [CLS] token embedding.
        hidden_size = backbone.config.hidden_size  # 768 for ViT-B/14
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

    # ── forward ──────────────────────────────────────────────────────────

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(logits, embeddings)``.

        ``logits``     — shape ``(B, num_classes)`` for cross-entropy.
        ``embeddings`` — shape ``(B, hidden_size)`` for contrastive loss.
        """
        outputs = self.backbone(pixel_values=pixel_values)
        cls_embedding = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.head(cls_embedding)
        return logits, cls_embedding

    # ── convenience ──────────────────────────────────────────────────────

    def print_trainable_parameters(self) -> None:
        """Print LoRA adapter statistics."""
        self.backbone.print_trainable_parameters()
