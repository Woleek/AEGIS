"""FaceNet (facenet-pytorch) model implementation for adversarial attacks.

Architecture: Inception-ResNet-v1.
Reference:    https://github.com/timesler/facenet-pytorch
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import onnxruntime as ort
from facenet_pytorch import InceptionResnetV1
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
    get_detector_asset_path,
    get_shared_insightface_detector,
    resolve_compute_device,
    warp_affine_pytorch,
)


# FaceNet was trained on MTCNN-style loose crops at 160×160 with no margin
# expansion. We keep that crop geometry but source the bbox from InsightFace
# RetinaFace (the detector used by every other FR model in this repo).
_FACENET_IMAGE_SIZE = 160
_FACENET_MARGIN = 0


_FACENET_SPECS: dict[str, VerificationModelSpec] = {
    "vggface2": VerificationModelSpec(
        name="facenet",
        variant="vggface2",
        display_name="FaceNet InceptionResnetV1 (VGGFace2)",
        embedding_dim=512,
        input_size=(160, 160),
        normalization="MTCNN fixed_image_standardization: (x - 127.5) / 128",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment=(
            "MTCNN-style extract_face crop geometry (margin=0, clamp, 160×160 resize) "
            "applied to the InsightFace bbox"
        ),
        threshold=0.498130,
        requirements=("facenet-pytorch", "insightface", "onnxruntime"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "facenet-vggface2.pt",
                description="FaceNet InceptionResnetV1 weights (VGGFace2)",
                source_url=(
                    "https://github.com/timesler/facenet-pytorch/releases/download/"
                    "v2.2.9/20180402-114759-vggface2.pt"
                ),
                alt_paths=(
                    MODELS_DIR / "facenet.pth",
                    MODELS_DIR / "facenet_vggface2.pt",
                    MODELS_DIR / "20180402-114759-vggface2.pt",
                    MODELS_DIR / "facenet" / "facenet_vggface2.pth",
                    Path("~/.cache/torch/checkpoints/20180402-114759-vggface2.pt"),
                ),
                install_hint=(
                    "Weights are auto-downloaded by facenet-pytorch on first use. "
                    "For local override, place the checkpoint under "
                    "models/facenet-vggface2.pt."
                ),
                auto_download=True,
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
            "Input to embed() must be RGB HWC tensor in [0, 1].",
            "Embeddings are 512-D and L2-normalized by InceptionResnetV1 in embedding mode.",
            "Cosine threshold is not provided by the upstream repo; calibrate with --threshold.",
        ),
    ),
    "casia-webface": VerificationModelSpec(
        name="facenet",
        variant="casia-webface",
        display_name="FaceNet InceptionResnetV1 (CASIA-WebFace)",
        embedding_dim=512,
        input_size=(160, 160),
        normalization="MTCNN fixed_image_standardization: (x - 127.5) / 128",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment=(
            "MTCNN-style extract_face crop geometry (margin=0, clamp, 160×160 resize) "
            "applied to the InsightFace bbox"
        ),
        threshold=None,
        requirements=("facenet-pytorch", "insightface", "onnxruntime"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "facenet-casia-webface.pt",
                description="FaceNet InceptionResnetV1 weights (CASIA-WebFace)",
                source_url=(
                    "https://github.com/timesler/facenet-pytorch/releases/download/"
                    "v2.2.9/20180408-102900-casia-webface.pt"
                ),
                alt_paths=(
                    MODELS_DIR / "facenet_casia-webface.pt",
                    MODELS_DIR / "20180408-102900-casia-webface.pt",
                    Path("~/.cache/torch/checkpoints/20180408-102900-casia-webface.pt"),
                ),
                install_hint=(
                    "Weights are auto-downloaded by facenet-pytorch on first use. "
                    "For local override, place the checkpoint under "
                    "models/facenet-casia-webface.pt."
                ),
                auto_download=True,
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
            "Input to embed() must be RGB HWC tensor in [0, 1].",
            "Embeddings are 512-D and L2-normalized by InceptionResnetV1 in embedding mode.",
            "Cosine threshold is not provided by the upstream repo; calibrate with --threshold.",
        ),
    ),
}


def get_facenet_specs() -> dict[str, VerificationModelSpec]:
    """Return the supported FaceNet variants."""

    return dict(_FACENET_SPECS)


class FaceNet:
    """FaceNet embedder backed by InsightFace RetinaFace + InceptionResnetV1.

    Embedding pipeline:
    1. Run InsightFace RetinaFace on the detached image to get the face box.
    2. Reproduce facenet-pytorch extract_face crop geometry (margin=0, clamp)
       with a differentiable affine warp to 160×160.
    3. Apply FaceNet fixed image standardization ((x*255 - 127.5) / 128).
    4. Forward through InceptionResnetV1 in embedding mode.
    5. Return (1, 512) embedding tensor.
    """

    spec: VerificationModelSpec
    device: torch.device

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        variant: Literal["vggface2", "casia-webface"] = "vggface2",
    ) -> None:
        self.spec = _FACENET_SPECS[variant]
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.ctx_id = 0 if self.device_str == "cuda" else -1
        self._variant = variant
        self._margin = _FACENET_MARGIN
        self._image_size = _FACENET_IMAGE_SIZE

        ort.set_default_logger_severity(3)
        self.validate_assets()
        self.detect_model: FaceDetector = get_shared_insightface_detector(self.ctx_id)
        checkpoint_path = self._resolve_checkpoint_path()
        try:
            if checkpoint_path is not None:
                state_dict = torch.load(checkpoint_path, map_location=self.device)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                if not isinstance(state_dict, dict):
                    raise ModelAssetMissingError(
                        f"FaceNet checkpoint is malformed: {checkpoint_path}"
                    )
                # num_classes=8631 matches DiffAM and the VGGFace2 training setup.
                # classify=False (default) ensures embeddings are returned, not logits.
                model = InceptionResnetV1(pretrained=None, num_classes=8631)
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                bad_missing = [key for key in missing if not key.startswith("logits.")]
                bad_unexpected = [key for key in unexpected if not key.startswith("logits.")]
                if bad_missing or bad_unexpected:
                    raise ModelAssetMissingError(
                        "FaceNet checkpoint is incompatible with InceptionResnetV1. "
                        f"Missing keys: {bad_missing[:5]} | Unexpected keys: {bad_unexpected[:5]}"
                    )
                self.model = model.eval().to(self.device)
            else:
                self.model = InceptionResnetV1(pretrained=variant).eval().to(self.device)
        except Exception as error:
            raise ModelAssetMissingError(
                "Failed to initialize FaceNet pretrained weights. "
                "Ensure network access for auto-download or place checkpoints manually."
            ) from error

    def _resolve_checkpoint_path(self) -> Path | None:
        assets = {asset.key: asset for asset in self.spec.assets}
        checkpoint_asset = assets["checkpoint"]
        for path in checkpoint_asset.candidate_paths():
            if path.exists():
                return path
        return None

    def validate_assets(self) -> None:
        assets = {asset.key: asset for asset in self.spec.assets}
        checkpoint_asset = assets["checkpoint"]
        if checkpoint_asset.auto_download:
            return
        for path in checkpoint_asset.candidate_paths():
            if path.exists():
                return
        raise ModelAssetMissingError(
            f"Missing FaceNet checkpoint for variant '{self.spec.variant}'."
        )

    @staticmethod
    def _to_numpy(image: torch.Tensor) -> np.ndarray:
        return (image.detach().cpu().clamp(0.0, 1.0).numpy() * 255.0).astype(np.float32)

    @staticmethod
    def _fixed_image_standardization(face_tensor: torch.Tensor) -> torch.Tensor:
        """Match facenet-pytorch fixed_image_standardization."""

        return (face_tensor - 127.5) / 128.0

    @staticmethod
    def _select_primary_box(boxes: np.ndarray | None) -> np.ndarray:
        """Select the first (highest-priority) detected face box."""

        if boxes is None:
            raise FaceNotDetectedError("No face detected in the image for FaceNet embedding.")

        box_array = np.asarray(boxes, dtype=np.float32)
        if box_array.size == 0:
            raise FaceNotDetectedError("No face detected in the image for FaceNet embedding.")
        if box_array.ndim == 1 and box_array.shape[0] >= 4:
            return box_array[:4]
        if box_array.ndim == 2 and box_array.shape[0] > 0 and box_array.shape[1] >= 4:
            return box_array[0, :4]

        raise ModelAssetMissingError(
            f"Unexpected detector output shape for FaceNet: {tuple(box_array.shape)}"
        )

    @staticmethod
    def _compute_mtcnn_extract_box(
        box: np.ndarray,
        image_hw: tuple[int, int],
        image_size: int,
        margin: int,
    ) -> tuple[int, int, int, int]:
        """Reproduce facenet-pytorch extract_face box expansion and clamping."""

        if image_size <= margin:
            raise ModelAssetMissingError(
                f"Invalid FaceNet crop settings: image_size={image_size}, margin={margin}."
            )

        box_arr = np.asarray(box, dtype=np.float32)
        if box_arr.shape[0] < 4:
            raise ModelAssetMissingError(
                f"Unexpected detector box shape for FaceNet: {tuple(box_arr.shape)}"
            )

        margin_x = margin * (box_arr[2] - box_arr[0]) / (image_size - margin)
        margin_y = margin * (box_arr[3] - box_arr[1]) / (image_size - margin)

        image_h, image_w = image_hw
        x1 = int(max(box_arr[0] - margin_x / 2.0, 0.0))
        y1 = int(max(box_arr[1] - margin_y / 2.0, 0.0))
        x2 = int(min(box_arr[2] + margin_x / 2.0, image_w))
        y2 = int(min(box_arr[3] + margin_y / 2.0, image_h))
        return (x1, y1, x2, y2)

    @staticmethod
    def _build_crop_resize_affine(
        crop_box: tuple[int, int, int, int],
        out_size: tuple[int, int],
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build an OpenCV-style affine matrix for crop+resize."""

        x1, y1, x2, y2 = crop_box
        crop_w = x2 - x1
        crop_h = y2 - y1
        if crop_w <= 0 or crop_h <= 0:
            raise FaceNotDetectedError(
                f"Invalid FaceNet crop box (empty region): {(x1, y1, x2, y2)}"
            )

        out_h, out_w = out_size
        sx = float(out_w) / float(crop_w)
        sy = float(out_h) / float(crop_h)

        # 0.5 offsets preserve the same crop+resize pixel-center convention as direct slicing.
        tx = -float(x1) * sx + 0.5 * (sx - 1.0)
        ty = -float(y1) * sy + 0.5 * (sy - 1.0)

        return torch.tensor(
            [[[sx, 0.0, tx], [0.0, sy, ty]]],
            dtype=dtype,
            device=device,
        )

    def _detect_primary_box(self, image: torch.Tensor) -> np.ndarray:
        detection = detect_face_for_embed(image, self.detect_model)
        if detection is None:
            raise FaceNotDetectedError("No face detected in the image for FaceNet embedding.")
        return detection.bbox

    def _extract_differentiable_face(self, image: torch.Tensor, box: np.ndarray) -> torch.Tensor:
        crop_box = self._compute_mtcnn_extract_box(
            box=box,
            image_hw=(int(image.shape[0]), int(image.shape[1])),
            image_size=self._image_size,
            margin=self._margin,
        )
        affine = self._build_crop_resize_affine(
            crop_box=crop_box,
            out_size=self.spec.input_size,
            device=self.device,
            dtype=torch.float32,
        )

        image_tensor = image.permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        face = warp_affine_pytorch(
            image_tensor=image_tensor,
            m_matrix=affine,
            out_size=self.spec.input_size,
        )
        # facenet-pytorch extract_face returns a float tensor in [0, 255].
        face = face * 255.0
        return self._fixed_image_standardization(face)

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        primary_box = self._detect_primary_box(image)
        face_crop = self._extract_differentiable_face(image, primary_box)
        embedding = self.model(face_crop)
        return embedding

    def embed_batch(
        self,
        images: list[torch.Tensor],
        detections: list[FaceDetection | None] | None = None,
    ) -> list[torch.Tensor | None]:
        if detections is not None and len(detections) != len(images):
            raise ValueError(
                f"detections length ({len(detections)}) must match images length ({len(images)})."
            )

        crops: list[torch.Tensor | None] = []
        for idx, image in enumerate(images):
            try:
                if detections is None:
                    primary_box = self._detect_primary_box(image)
                else:
                    det = detections[idx]
                    if det is None:
                        crops.append(None)
                        continue
                    primary_box = det.bbox
                crops.append(self._extract_differentiable_face(image, primary_box))
            except FaceNotDetectedError:
                crops.append(None)

        valid_indices = [i for i, c in enumerate(crops) if c is not None]
        if not valid_indices:
            return [None] * len(images)

        batch = torch.cat([crops[i] for i in valid_indices])
        embeddings = self.model(batch)  # (N, 512); InceptionResnetV1 L2-normalizes in embedding mode

        results: list[torch.Tensor | None] = [None] * len(images)
        for out_idx, src_idx in enumerate(valid_indices):
            results[src_idx] = embeddings[out_idx : out_idx + 1]
        return results

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.embed(image)
