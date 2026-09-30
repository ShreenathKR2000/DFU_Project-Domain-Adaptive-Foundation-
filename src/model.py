"""
DINOv2 ViT-B/14 backbone with LoRA adapters and a 4-class head.

Trainable parameters: LoRA adapters (Q/V projections) + classification head.
Everything else in the backbone stays frozen.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from transformers import Dinov2Model


LABEL_NAMES = ["none", "infection", "ischaemia", "both"]
NUM_CLASSES = len(LABEL_NAMES)


def _lora_target_modules(backbone: nn.Module) -> list[str]:
    """Q/V projection names: ``query``/``value`` (transformers 4.x) or
    ``q_proj``/``v_proj`` (transformers 5.x)."""
    names = {n.rsplit(".", 1)[-1] for n, _ in backbone.named_modules()}
    return ["query", "value"] if "query" in names else ["q_proj", "v_proj"]


class DFUDinoLoRA(nn.Module):
    """Frozen DINOv2-ViT-B/14 + LoRA adapters on Q/V projections + linear head.

    Parameters
    ----------
    model_name : str
        Hugging Face model id (default ``facebook/dinov2-base``).
    num_classes : int
        Number of output classes (default 4).
    lora_rank : int
        LoRA low-rank dimension (default 16).
    lora_alpha : int
        LoRA scaling factor (default 32).
    lora_dropout : float
        Dropout inside LoRA layers (default 0.1).
    pretrained_lora_path : str or None
        Path to a Phase-1 LoRA checkpoint (``dfu_pretrained_backbone.pt``).
        When provided, the saved LoRA adapter weights are loaded into the
        backbone *before* the classification head is trained, giving the
        adapters a domain-adapted starting point.
    """

    def __init__(
        self,
        model_name = "facebook/dinov2-base",
        num_classes: int = NUM_CLASSES,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        pretrained_lora_path: Optional[str] = None,
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
            target_modules=_lora_target_modules(backbone),
            bias="none",
        )
        self.backbone = get_peft_model(backbone, lora_config)

        # 2b. (Optional) Load domain-adapted LoRA weights from Phase 1.
        if pretrained_lora_path is not None:
            state = torch.load(pretrained_lora_path, map_location="cpu")
            lora_state = {k: v for k, v in state.items() if "lora_" in k}
            n_lora = len(lora_state)
            if n_lora == 0:
                raise ValueError(f"No LoRA tensors found in {pretrained_lora_path}")
            # pretrain.py saves backbone.state_dict(), whose keys already match
            # this model, so load them directly.  (Older peft versions'
            # set_peft_model_state_dict re-inserts the adapter name and then
            # silently ignores every key.)  Fall back to it for checkpoints
            # saved with get_peft_model_state_dict.
            result = self.backbone.load_state_dict(lora_state, strict=False)
            if result.unexpected_keys:
                set_peft_model_state_dict(self.backbone, state)
            # LoRA B matrices start at zero, so if they are still all zero the
            # load was silently ignored and the run would equal "scratch".
            n_nonzero = sum(
                1 for n, p in self.backbone.named_parameters()
                if "lora_B" in n and p.abs().sum() > 0
            )
            if n_nonzero == 0:
                raise RuntimeError(
                    f"Phase-1 LoRA weights from {pretrained_lora_path} were not applied."
                )
            print(
                f"Loaded Phase-1 LoRA weights from {pretrained_lora_path} "
                f"({n_lora} LoRA tensors, {n_nonzero} non-zero lora_B matrices)"
            )

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
