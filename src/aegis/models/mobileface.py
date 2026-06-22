"""MobileFaceNet face recognition model for adversarial attacks.

Architecture: MobileFaceNet (512-D embedding).
Reference:    DiffAM / TreB1eN InsightFace_Pytorch
Checkpoint:   models/mobile_face.pth
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from insightface.utils.face_align import estimate_norm
import torch
import torch.nn as nn

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
from ._irse_backbone import MobileFaceNetBackbone


_MOBILEFACE_SPECS: dict[str, VerificationModelSpec] = {
    "default": VerificationModelSpec(
        name="mobileface",
        variant="default",
        display_name="MobileFaceNet (512-D)",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x / 1 - 0.5) / 0.5  [RGB, float32, i.e. [0,1]→[-1,1]]",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=0.351954,
        requirements=("insightface",),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "mobile_face.pth",
                description="MobileFaceNet pretrained weights",
                install_hint=(
                    "Place the pretrained checkpoint at models/mobile_face.pth. "
                    "Compatible with DiffAM assets/models/mobile_face.pth."
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
        ),
    ),
}


def get_mobileface_specs() -> dict[str, VerificationModelSpec]:
    return dict(_MOBILEFACE_SPECS)


def _load_mobileface_backbone(checkpoint_path: Path, device: str) -> MobileFaceNetBackbone:
    model = MobileFaceNetBackbone(embedding_size=512)
    state_dict = torch.load(checkpoint_path, weights_only=False, map_location=torch.device(device))
    if not isinstance(state_dict, dict):
        raise ModelAssetMissingError(
            f"MobileFaceNet checkpoint is not a state dict: {checkpoint_path}"
        )
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model


class MobileFace:
    """MobileFaceNet embedder for differentiable face verification.

    Embedding pipeline (fully differentiable through the backbone):
    1. Detect face keypoints via InsightFace buffalo_l (non-differentiable, detached).
    2. Warp-align to 112×112 via differentiable affine grid sampling.
    3. Normalize: (x - 0.5) / 0.5  (RGB [0,1] → [-1,1]).
    4. Forward through MobileFaceNetBackbone → 512-D features.
    5. L2-normalize the output.
    """

    spec: VerificationModelSpec
    device: torch.device

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        model_path: Path,
        variant: Literal["default"] = "default",
    ) -> None:
        self.spec = _MOBILEFACE_SPECS[variant]
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.ctx_id = 0 if self.device_str == "cuda" else -1

        self.validate_assets()
        self.detect_model: FaceDetector = get_shared_insightface_detector(self.ctx_id)

        assets = {a.key: a for a in self.spec.assets}
        checkpoint_path = ensure_asset_present(assets["checkpoint"])
        self.model: MobileFaceNetBackbone = _load_mobileface_backbone(
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
            raise FaceNotDetectedError("No face detected for MobileFaceNet embedding.")
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
