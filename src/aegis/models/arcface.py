"""ArcFace face recognition model implementation for adversarial attacks.

Architecture: IResNet (InsightFace arcface_torch).
Reference: https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from insightface.utils.face_align import estimate_norm
import torch
import torch.nn as nn
import torch.nn.functional as F

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

_ARCFACE_SPECS: dict[str, VerificationModelSpec] = {
    "r50": VerificationModelSpec(
        name="arcface",
        variant="r50",
        display_name="ArcFace IResNet-50",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x / 255 - 0.5) / 0.5  [RGB, float32]",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=0.239179,
        requirements=("insightface", "torchvision"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "ms1mv3_arcface_r50" / "backbone.pth",
                description="ArcFace IResNet-50 backbone trained on MS1MV3",
                source_url=(
                    "https://github.com/deepinsight/insightface/tree/master/"
                    "recognition/arcface_torch#model-zoo"
                ),
                alt_paths=(MODELS_DIR / "arcface_ir50_ms1mv3.pth",),
                install_hint=(
                    "Download the official ArcFace Torch MS1MV3 r50 model and place it under "
                    "models/ms1mv3_arcface_r50/backbone.pth. The legacy flat filename "
                    "models/arcface_ir50_ms1mv3.pth is also accepted."
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
            "Threshold is not set; calibrate with --threshold or use --embedder adaface.",
            "Rendered images and reference images must be RGB HWC tensors in [0, 1].",
            "Output embeddings are L2-normalized to unit length.",
        ),
    ),
    "r100": VerificationModelSpec(
        name="arcface",
        variant="r100",
        display_name="ArcFace IResNet-100",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x / 255 - 0.5) / 0.5  [RGB, float32]",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=None,
        requirements=("insightface", "torchvision"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "ms1mv3_arcface_r100" / "backbone.pth",
                description="ArcFace IResNet-100 backbone trained on MS1MV3",
                source_url=(
                    "https://github.com/deepinsight/insightface/tree/master/"
                    "recognition/arcface_torch#model-zoo"
                ),
                alt_paths=(MODELS_DIR / "arcface_r100_ms1mv3.pth",),
                install_hint=(
                    "Download the official ArcFace Torch MS1MV3 r100 model and place it under "
                    "models/ms1mv3_arcface_r100/backbone.pth. The legacy flat filename "
                    "models/arcface_r100_ms1mv3.pth is also accepted."
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
            "Threshold is not set; calibrate with --threshold or use --embedder adaface.",
            "Rendered images and reference images must be RGB HWC tensors in [0, 1].",
            "Output embeddings are L2-normalized to unit length.",
        ),
    ),
}


# ---------------------------------------------------------------------------
# IResNet backbone — matches InsightFace arcface_torch exactly so pretrained
# state dicts load without key remapping.
# Reference: deepinsight/insightface/recognition/arcface_torch/backbones/iresnet.py
# ---------------------------------------------------------------------------

def _conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)


def _conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, 1, stride=stride, bias=False)


class _IBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None,
                 groups: int = 1, base_width: int = 64, dilation: int = 1) -> None:
        super().__init__()
        if groups != 1 or base_width != 64:
            raise ValueError("_IBasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported")
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-5)
        self.conv1 = _conv3x3(inplanes, planes)
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-5)
        self.prelu = nn.PReLU(planes)
        self.conv2 = _conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-5)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)
        out = self.bn3(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        return out


class _IResNet(nn.Module):
    """IResNet backbone for ArcFace.

    Produces 512-D BN1d features; call `.embed()` on the wrapper for
    L2-normalized output.
    """

    fc_scale: int = 7 * 7

    def __init__(self, layers: list[int], num_features: int = 512,
                 dropout: float = 0.0) -> None:
        super().__init__()
        self.inplanes = 64
        self.dilation = 1
        self.groups = 1
        self.base_width = 64

        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64, eps=1e-5)
        self.prelu = nn.PReLU(64)

        self.layer1 = self._make_layer(_IBasicBlock, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(_IBasicBlock, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(_IBasicBlock, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(_IBasicBlock, 512, layers[3], stride=2)

        self.bn2 = nn.BatchNorm2d(512, eps=1e-5)
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(512 * self.fc_scale, num_features)
        self.features = nn.BatchNorm1d(num_features, eps=1e-5)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

    def _make_layer(self, block: type[_IBasicBlock], planes: int, num_blocks: int,
                    stride: int = 1) -> nn.Sequential:
        downsample: nn.Module | None = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                _conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-5),
            )
        layers: list[nn.Module] = [
            block(self.inplanes, planes, stride, downsample,
                  self.groups, self.base_width, self.dilation)
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(block(self.inplanes, planes,
                                groups=self.groups, base_width=self.base_width,
                                dilation=self.dilation))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.prelu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.bn2(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        x = self.features(x)
        return x


_IRESNET_LAYERS: dict[str, list[int]] = {
    "r50": [3, 4, 14, 3],
    "r100": [3, 13, 30, 3],
}


def _load_iresnet(
    checkpoint_path: Path,
    variant: Literal["r50", "r100"],
    device: str,
) -> _IResNet:
    layers = _IRESNET_LAYERS[variant]
    model = _IResNet(layers=layers)

    state_dict = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location=torch.device(device),
    )
    if not isinstance(state_dict, dict):
        raise ModelAssetMissingError(
            f"ArcFace checkpoint did not load as a state dict: {checkpoint_path}."
        )
    # Accept both bare state dicts and wrapped {'state_dict': ...} formats.
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    model.load_state_dict(state_dict)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# ArcFace embedder
# ---------------------------------------------------------------------------

def get_arcface_specs() -> dict[str, VerificationModelSpec]:
    """Return the supported ArcFace variants."""
    return dict(_ARCFACE_SPECS)


class ArcFace:
    """ArcFace embedding model for differentiable face verification.

    Embedding pipeline:
    1. Detect face keypoints (InsightFace buffalo_l, non-differentiable).
    2. Warp-align to 112x112 via differentiable affine grid sampling.
    3. Normalize: (x - 0.5) / 0.5  (RGB [0, 1] → [-1, 1]).
    4. Forward through IResNet backbone.
    5. L2-normalize the 512-D output.
    """

    spec: VerificationModelSpec
    device: torch.device

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        model_path: Path,
        variant: Literal["r50", "r100"] = "r50",
    ) -> None:
        self.spec = _ARCFACE_SPECS[variant]
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.ctx_id = 0 if self.device_str == "cuda" else -1
        self._variant = variant

        self.validate_assets()
        self.detect_model: FaceDetector = get_shared_insightface_detector(self.ctx_id)

        assets = {a.key: a for a in self.spec.assets}
        checkpoint_path = ensure_asset_present(assets["checkpoint"])
        self.model: _IResNet = _load_iresnet(
            checkpoint_path, variant, self.device_str
        ).to(self.device)
        self.model.eval()

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
            raise FaceNotDetectedError("No face detected in the image for ArcFace embedding.")
        return self._align_from_detection(image, detection)

    def _normalize(self, aligned: torch.Tensor) -> torch.Tensor:
        """(C, H, W) RGB [0,1] → (1, C, H, W) normalized to [−1, 1]."""
        return aligned.sub(0.5).div(0.5).unsqueeze(0).to(self.device)

    @staticmethod
    def _l2_norm(vec: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vec, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return vec / norm

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        """Embed a single (H, W, C) RGB [0,1] tensor. Returns (1, 512) L2-normalized."""
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
        embeddings = self._l2_norm(self.model(batch))  # (N, 512)

        results: list[torch.Tensor | None] = [None] * len(images)
        for out_idx, src_idx in enumerate(valid_indices):
            results[src_idx] = embeddings[out_idx : out_idx + 1]
        return results

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.embed(image)
