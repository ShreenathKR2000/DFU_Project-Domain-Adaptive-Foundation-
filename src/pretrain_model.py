"""
SimMIM-style Masked Image Modeling wrapper around DFUDinoLoRA.

During domain-adaptive pre-training (Phase 1), random patches of the input
image are masked (zeroed out), the LoRA-injected DINOv2 encoder processes the
corrupted image, and a lightweight linear decoder reconstructs the original
pixel values of the masked patches.

Trainable parameters: LoRA adapters (Q/V) + decoder head.
The classification head inside DFUDinoLoRA is frozen (unused here).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model import DFUDinoLoRA


class DFUDinoLoRAForMIM(nn.Module):
    """Frozen DINOv2 + LoRA backbone with a patch-reconstruction decoder.

    Parameters
    ----------
    mask_ratio : float
        Fraction of patches to mask each forward pass (default 0.6).
    patch_size : int
        ViT patch size in pixels (default 14 for DINOv2 ViT-B/14).
    loss_fn : str
        ``"l1"`` (default) or ``"mse"`` for reconstruction loss.
    **backbone_kwargs
        Forwarded to :class:`DFUDinoLoRA` (e.g. ``lora_rank``,
        ``lora_alpha``, ``model_name``).
    """

    def __init__(
        self,
        mask_ratio: float = 0.6,
        patch_size: int = 14,
        loss_fn: str = "l1",
        **backbone_kwargs,
    ) -> None:
        super().__init__()

        # ── encoder (reuses the full DINOv2 + LoRA setup) ───────────────
        self.encoder = DFUDinoLoRA(**backbone_kwargs)

        # Freeze the classification head — it's unused during pre-training.
        for param in self.encoder.head.parameters():
            param.requires_grad = False

        # ── decoder (lightweight: LayerNorm + Linear → pixel patch) ─────
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        hidden_size = 768  # ViT-B hidden dimension
        decoder_dim = patch_size * patch_size * 3  # RGB values per patch

        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, decoder_dim),
        )

        self.loss_fn = F.l1_loss if loss_fn == "l1" else F.mse_loss

    # ── masking helpers ──────────────────────────────────────────────────

    @torch.no_grad()
    def _generate_mask(
        self, batch_size: int, num_patches: int, device: torch.device
    ) -> torch.BoolTensor:
        """Return ``(B, N)`` boolean mask where ``True`` = masked."""
        return torch.rand(batch_size, num_patches, device=device) < self.mask_ratio

    def _apply_pixel_mask(
        self, pixel_values: torch.Tensor, mask: torch.BoolTensor
    ) -> torch.Tensor:
        """Zero out image regions corresponding to masked patches."""
        B, C, H, W = pixel_values.shape
        P = self.patch_size
        pH, pW = H // P, W // P

        # (B, pH*pW) → (B, 1, H, W) via reshape + nearest upsample
        pixel_mask = (
            mask
            .reshape(B, pH, pW)
            .unsqueeze(1)                          # (B, 1, pH, pW)
            .repeat_interleave(P, dim=2)           # (B, 1,  H, pW)
            .repeat_interleave(P, dim=3)           # (B, 1,  H,  W)
        )
        return pixel_values.masked_fill(pixel_mask, 0.0)

    def _patchify(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Reshape ``(B, 3, H, W)`` → ``(B, num_patches, patch_size²×3)``."""
        B, C, H, W = pixel_values.shape
        P = self.patch_size
        pH, pW = H // P, W // P

        x = pixel_values.reshape(B, C, pH, P, pW, P)
        x = x.permute(0, 2, 4, 3, 5, 1)          # (B, pH, pW, P, P, C)
        x = x.reshape(B, pH * pW, -1)             # (B, N, P²·C)
        return x

    # ── forward ──────────────────────────────────────────────────────────

    def forward(
        self, pixel_values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.BoolTensor]:
        """Run one MIM forward pass.

        Returns
        -------
        loss : scalar tensor
            Reconstruction loss computed **only** on the masked patches.
        pred_patches : ``(B, N, P²×3)``
            Decoded pixel values for every patch.
        mask : ``(B, N)``
            Boolean mask (``True`` = masked / reconstructed).
        """
        B, C, H, W = pixel_values.shape
        P = self.patch_size
        num_patches = (H // P) * (W // P)

        # 1. Random masking → corrupt input
        mask = self._generate_mask(B, num_patches, pixel_values.device)
        masked_images = self._apply_pixel_mask(pixel_values, mask)

        # 2. Encode corrupted image (all 256 patch tokens + CLS)
        outputs = self.encoder.backbone(pixel_values=masked_images)
        patch_tokens = outputs.last_hidden_state[:, 1:]   # drop [CLS]

        # 3. Decode patch tokens → predicted pixel values
        pred_patches = self.decoder(patch_tokens)          # (B, N, P²×3)

        # 4. Reconstruction targets from the *original* image
        target_patches = self._patchify(pixel_values)      # (B, N, P²×3)

        # 5. Loss on masked patches only
        loss = self.loss_fn(pred_patches[mask], target_patches[mask])

        return loss, pred_patches, mask

    # ── convenience ──────────────────────────────────────────────────────

    def print_trainable_parameters(self) -> None:
        """Print LoRA adapter + decoder statistics."""
        self.encoder.print_trainable_parameters()
        dec_params = sum(p.numel() for p in self.decoder.parameters())
        print(f"decoder params : {dec_params:,}")
