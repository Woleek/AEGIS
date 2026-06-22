"""IRSE-50 face recognition model for adversarial attacks.

Architecture: IResNet-50 IR-SE mode (512-D embedding).
Reference:    DiffAM / TreB1eN InsightFace_Pytorch (assets/models/irse.py)
Checkpoint:   models/irse50.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from insightface.utils.face_align import estimate_norm
import torch

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
from ._irse_backbone import IrseBackbone


_IRSE50_SPECS: dict[str, VerificationModelSpec] = {
    "ir_se50": VerificationModelSpec(
        name="irse50",
        variant="ir_se50",
        display_name="IRSE-50",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x - 0.5) / 0.5  [RGB, float32, i.e. [0,1]→[-1,1]]",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=0.329331,
        requirements=("insightface",),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "irse50.pth",
                description="IRSE-50 pretrained weights",
                install_hint=(
                    "Place the pretrained checkpoint at models/irse50.pth. "
                    "Compatible with DiffAM assets/models/irse50.pth."
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
            "Threshold is not calibrated; use --threshold to set one.",
            "Rendered images and reference images must be RGB HWC tensors in [0, 1].",
            "Output embeddings are L2-normalized to unit length.",
            "drop_ratio=0.6 matches DiffAM irse.Backbone(50, 0.6, 'ir_se').",
        ),
    ),
}


def get_irse50_specs() -> dict[str, VerificationModelSpec]:
    return dict(_IRSE50_SPECS)


def _load_irse50_backbone(checkpoint_path: Path, device: str) -> IrseBackbone:
    # drop_ratio=0.6 matches DiffAM model_utils.py: irse.Backbone(50, 0.6, 'ir_se').
    model = IrseBackbone(num_layers=50, drop_ratio=0.6, mode="ir_se")
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location=torch.device(device))
    if not isinstance(state_dict, dict):
        raise ModelAssetMissingError(
            f"IRSE-50 checkpoint is not a state dict: {checkpoint_path}"
        )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


class IRSE50:
    """IRSE-50 embedder for differentiable face verification.

    Embedding pipeline (fully differentiable through the backbone):
    1. Detect face keypoints via InsightFace buffalo_l (non-differentiable, detached).
    2. Warp-align to 112×112 via differentiable affine grid sampling.
    3. Normalize: (x - 0.5) / 0.5  (RGB [0,1] → [-1,1]).
    4. Forward through IrseBackbone (50-layer IR-SE) → 512-D BN features.
    5. L2-normalize the output.
    """

    spec: VerificationModelSpec
    device: torch.device

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        model_path: Path,
        variant: Literal["ir_se50"] = "ir_se50",
    ) -> None:
        self.spec = _IRSE50_SPECS[variant]
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.ctx_id = 0 if self.device_str == "cuda" else -1

        self.validate_assets()
        self.detect_model: FaceDetector = get_shared_insightface_detector(self.ctx_id)

        assets = {a.key: a for a in self.spec.assets}
        checkpoint_path = ensure_asset_present(assets["checkpoint"])
        self.model: IrseBackbone = _load_irse50_backbone(
            checkpoint_path, self.device_str
        ).to(self.device)

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
            raise FaceNotDetectedError("No face detected for IRSE-50 embedding.")
        return self._align_from_detection(image, detection)

    def _normalize(self, aligned: torch.Tensor) -> torch.Tensor:
        return aligned.sub(0.5).div(0.5).unsqueeze(0).to(self.device)

    @staticmethod
    def _l2_norm(vec: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vec, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return vec / norm

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        aligned = self._align_face(image)
        tensor = self._normalize(aligned)
        features = self.model(tensor)
        return self._l2_norm(features)

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
        embeddings = self._l2_norm(self.model(batch))

        results: list[torch.Tensor | None] = [None] * len(images)
        for out_idx, src_idx in enumerate(valid_indices):
            results[src_idx] = embeddings[out_idx : out_idx + 1]
        return results

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.embed(image)
