"""SwinFace face recognition model implementation for adversarial attacks.

Architecture: Swin Transformer Tiny.
Reference:    https://github.com/lxq1000/SwinFace
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import timm
import torch
import torch.nn as nn
from insightface.utils.face_align import estimate_norm

from ..config import MODELS_DIR
from .base import (
    FaceDetection,
    FaceDetector,
    FaceNotDetectedError,
    ModelAssetMissingError,
    ModelAssetSpec,
    VerificationModelSpec,
    detect_face_for_embed,
    ensure_asset_present,
    get_detector_asset_path,
    get_shared_insightface_detector,
    resolve_compute_device,
    warp_affine_pytorch,
)


# ---------------------------------------------------------------------------
# Model specs
# ---------------------------------------------------------------------------

_SWINFACE_SPECS: dict[str, VerificationModelSpec] = {
    "swin_t": VerificationModelSpec(
        name="swinface",
        variant="swin_t",
        display_name="SwinFace Swin-T",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x / 255 - 0.5) / 0.5  [RGB, float32]",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=0.256976,
        requirements=("insightface", "timm"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "swinface_swin_t.pth",
                description="SwinFace Swin-T backbone checkpoint",
                alt_paths=(MODELS_DIR / "swinface" / "checkpoint_step_79999_gpu_0.pt",),
                source_url=(
                    "https://drive.google.com/drive/folders/"
                    "1NjVN3Kp_Tmwt17hWCIWgHpuWzkHYaman"
                ),
                install_hint=(
                    "Download the SwinFace checkpoint from the official Google Drive. "
                    "Both the full training checkpoint (with 'state_dict_backbone' key) "
                    "and a pre-extracted backbone state dict are accepted. "
                    "Place it at models/swinface_swin_t.pth."
                ),
            ),
            ModelAssetSpec(
                key="detector",
                path=get_detector_asset_path(),
                description="InsightFace buffalo_l detector",
                install_hint=(
                    "Run the model once or pre-install with "
                    "`python -c \"import insightface; "
                    "insightface.utils.ensure_available('models', 'buffalo_l', "
                    "root='~/.insightface')\"`."
                ),
                auto_download=True,
            ),
        ),
        notes=(
            "Threshold is not set; calibrate with --threshold.",
            "Rendered images and reference images must be RGB HWC tensors in [0, 1].",
            "Output embeddings are L2-normalized to unit length.",
            "Backbone: Swin-T patch_size=2, img_size=112; embedding via FC-BN-FC-BN 'feature' subnet.",
        ),
    ),
}


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def _extract_backbone_state_dict(raw: object, checkpoint_path: Path) -> dict:
    """Return a flat backbone state dict from a raw checkpoint object.

    Accepts:
    - A full SwinFace training checkpoint dict with ``state_dict_backbone``.
    - A bare backbone state dict (all values are ``torch.Tensor``).
    """
    if not isinstance(raw, dict):
        raise ModelAssetMissingError(
            f"SwinFace checkpoint is not a dict: {checkpoint_path}."
        )
    if "state_dict_backbone" in raw:
        return raw["state_dict_backbone"]
    non_tensor = [k for k, v in raw.items() if not isinstance(v, torch.Tensor)]
    if non_tensor:
        raise ModelAssetMissingError(
            f"SwinFace checkpoint has unexpected format: {checkpoint_path}. "
            f"Expected 'state_dict_backbone' key or a bare state dict, "
            f"but found non-tensor keys: {non_tensor[:5]}."
        )
    return raw


def _remap_checkpoint_to_timm(sd: dict) -> dict:
    """Remap SwinFace checkpoint keys to match timm's Swin-T key naming.

    The original Swin Transformer code (used by SwinFace) attaches each
    ``PatchMerging`` module at the END of its stage:
        layers.N.downsample  →  applied after layers.N.blocks

    timm's SwinTransformer attaches each ``PatchMerging`` at the beginning
    of the next stage (as an input projection):
        layers.(N+1).downsample  →  applied before layers.(N+1).blocks

    The result is that block weights (``layers.N.blocks.*``) are indexed the
    same in both implementations, but downsample weights are shifted by one:
        checkpoint layers.N.downsample.*  →  timm layers.(N+1).downsample.*
    """
    remapped: dict = {}
    for k, v in sd.items():
        if k.startswith("head.") or k.startswith("feature."):
            continue
        if ".downsample." in k:
            parts = k.split(".")
            n = int(parts[1])
            new_k = ".".join(["layers", str(n + 1)] + parts[2:])
            remapped[new_k] = v
        else:
            remapped[k] = v
    return remapped


def _load_swinface_modules(
    checkpoint_path: Path,
    device: str,
) -> tuple[nn.Module, nn.Module]:
    """Load and return ``(backbone, feature)`` from the checkpoint.

    backbone
        timm Swin-T (``patch_size=2``, ``img_size=112``, ``num_classes=0``).
        ``forward_features(x)`` returns ``[B, T, 768]`` token sequence (T=49);
        global-average-pool to ``[B, 768]`` is performed in ``embed()``.

    feature
        ``nn.Sequential`` FC-BN-FC-BN face recognition subnet:
        Linear(768, 768) → BN(768) → Linear(768, 512) → BN(512).
        Confirmed from checkpoint ``feature.{0..3}`` keys.
    """
    try:
        raw = torch.load(
            checkpoint_path,
            weights_only=False,
            map_location=torch.device(device),
        )
    except Exception as error:
        raise ModelAssetMissingError(
            f"Failed to load SwinFace checkpoint from {checkpoint_path}: {error}"
        ) from error

    sd = _extract_backbone_state_dict(raw, checkpoint_path)

    # --- Backbone ---
    backbone = timm.create_model(
        "swin_tiny_patch4_window7_224",
        pretrained=False,
        img_size=112,
        patch_size=2,
        num_classes=0,
    )
    backbone_sd = _remap_checkpoint_to_timm(sd)
    missing, unexpected = backbone.load_state_dict(backbone_sd, strict=False)

    # Tolerate attn_mask and relative_position_index buffer presence/absence —
    # timm computes these deterministically at runtime.
    _buf_names = {"attn_mask", "relative_position_index"}
    bad_missing = [k for k in missing if not any(b in k for b in _buf_names)]
    bad_unexpected = [k for k in unexpected if not any(b in k for b in _buf_names)]
    if bad_missing or bad_unexpected:
        raise ModelAssetMissingError(
            f"SwinFace backbone state dict mismatch after key remapping.\n"
            f"  Missing   : {bad_missing[:5]}\n"
            f"  Unexpected: {bad_unexpected[:5]}\n"
            "Ensure the checkpoint is from the official SwinFace repo "
            "(state_dict_backbone with patch_size=2 / img_size=112)."
        )
    backbone.eval()

    # --- Feature module (FC-BN-FC-BN) ---
    feature = nn.Sequential(
        nn.Linear(768, 768, bias=False),
        nn.BatchNorm1d(768),
        nn.Linear(768, 512, bias=False),
        nn.BatchNorm1d(512),
    )
    feature_sd = {
        k[len("feature."):]: v for k, v in sd.items() if k.startswith("feature.")
    }
    if not feature_sd:
        raise ModelAssetMissingError(
            f"SwinFace checkpoint is missing 'feature.*' recognition subnet keys: "
            f"{checkpoint_path}."
        )
    feature.load_state_dict(feature_sd, strict=True)
    feature.eval()

    return backbone, feature


class _SwinFaceNet(nn.Module):
    """Swin-T backbone + FC-BN-FC-BN recognition head.
    """

    def __init__(self, backbone: nn.Module, feature: nn.Module) -> None:
        super().__init__()
        self._backbone = backbone
        self._feature = feature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # timm SwinTransformer.forward_features returns [B, H, W, C].
        forward_features = getattr(self._backbone, "forward_features")
        tokens: torch.Tensor = forward_features(x)
        pooled = tokens.mean(dim=(1, 2))  # [B, 768]
        return self._feature(pooled)      # [B, 512]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------

def get_swinface_specs() -> dict[str, VerificationModelSpec]:
    """Return the supported SwinFace variants."""
    return dict(_SWINFACE_SPECS)


# ---------------------------------------------------------------------------
# SwinFace embedder
# ---------------------------------------------------------------------------

class SwinFace:
    """SwinFace embedding model for differentiable face verification.

    Embedding pipeline:
    1. Detect face keypoints (InsightFace buffalo_l, non-differentiable).
    2. Warp-align to 112x112 via differentiable affine grid sampling.
    3. Normalize: (x - 0.5) / 0.5  (RGB [0, 1] -> [-1, 1]).
    4. Forward through _SwinFaceNet (Swin-T + FC-BN-FC-BN) -> [B, 512].
    5. L2-normalize the 512-D output.
    """

    spec: VerificationModelSpec
    device: torch.device
    model: _SwinFaceNet

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        model_path: Path,
        variant: Literal["swin_t"] = "swin_t",
    ) -> None:
        self.spec = _SWINFACE_SPECS[variant]
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.ctx_id = 0 if self.device_str == "cuda" else -1

        self.validate_assets()
        self.detect_model: FaceDetector = get_shared_insightface_detector(self.ctx_id)

        assets = {a.key: a for a in self.spec.assets}
        checkpoint_path = ensure_asset_present(assets["checkpoint"])
        backbone, feature = _load_swinface_modules(checkpoint_path, self.device_str)
        self.model = _SwinFaceNet(backbone, feature).to(self.device)

    def validate_assets(self) -> None:
        assets = {a.key: a for a in self.spec.assets}
        ensure_asset_present(assets["checkpoint"])

    def _align_from_detection(
        self,
        image: torch.Tensor,
        detection: FaceDetection,
    ) -> torch.Tensor:
        M = estimate_norm(detection.landmarks, image_size=self.spec.input_size[0])
        aligned = warp_affine_pytorch(
            image_tensor=image.permute(2, 0, 1).unsqueeze(0).float(),
            m_matrix=torch.from_numpy(M).unsqueeze(0).to(image.device).float(),
            out_size=self.spec.input_size,
        )
        return aligned.squeeze(0)  # (C, H, W)

    def _align_face(self, image: torch.Tensor) -> torch.Tensor:
        detection = detect_face_for_embed(image, self.detect_model)
        if detection is None:
            raise FaceNotDetectedError("No face detected in the image for SwinFace embedding.")
        return self._align_from_detection(image, detection)

    def _normalize(self, aligned: torch.Tensor) -> torch.Tensor:
        """(C, H, W) RGB [0,1] -> (1, C, H, W) normalized to [-1, 1]."""
        return aligned.sub(0.5).div(0.5).unsqueeze(0).to(self.device)

    @staticmethod
    def _l2_norm(vec: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vec, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return vec / norm

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        """Embed a single (H, W, C) RGB [0,1] tensor. Returns (1, 512) L2-normalized."""
        aligned = self._align_face(image)
        tensor = self._normalize(aligned)
        embedding = self.model(tensor)
        return self._l2_norm(embedding)

    def embed_batch(
        self,
        images: list[torch.Tensor],
        detections: list[FaceDetection | None] | None = None,
    ) -> list[torch.Tensor | None]:
        if detections is not None and len(detections) != len(images):
            raise ValueError(
                f"detections length ({len(detections)}) must match images length ({len(images)})."
            )

        aligned: list[torch.Tensor | None] = []
        for idx, image in enumerate(images):
            try:
                if detections is None:
                    aligned.append(self._align_face(image))
                else:
                    det = detections[idx]
                    if det is None:
                        aligned.append(None)
                    else:
                        aligned.append(self._align_from_detection(image, det))
            except FaceNotDetectedError:
                aligned.append(None)

        valid_indices = [i for i, a in enumerate(aligned) if a is not None]
        if not valid_indices:
            return [None] * len(images)

        batch = torch.cat([self._normalize(aligned[i]) for i in valid_indices])
        embeddings = self._l2_norm(self.model(batch))  # (N, 512)

        results: list[torch.Tensor | None] = [None] * len(images)
        for out_idx, src_idx in enumerate(valid_indices):
            results[src_idx] = embeddings[out_idx : out_idx + 1]
        return results

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.embed(image)
