"""AdaFace face recognition model implementation for adversarial attacks."""

from __future__ import annotations

from collections import namedtuple
from pathlib import Path
from typing import Literal

from insightface.utils.face_align import estimate_norm
import onnxruntime as ort
import torch
import torch.nn as nn
from torchvision import transforms

from ..config import MODELS_DIR
from .base import (
    FaceDetection,
    FaceDetector,
    FaceEmbedder,
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


_ADAFACE_SPECS: dict[str, VerificationModelSpec] = {
    "ir50": VerificationModelSpec(
        name="adaface",
        variant="ir50",
        display_name="AdaFace IR-50",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x / 255 - 0.5) / 0.5",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=0.197146,
        requirements=("insightface", "onnxruntime", "torchvision"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "adaface_ir50_ms1mv2.ckpt",
                description="AdaFace IR-50 checkpoint",
                source_url=(
                    "https://drive.google.com/file/d/"
                    "1eUaSHG4pGlIZK7hBkqjyp2fc2epKoBvI/view?usp=sharing"
                ),
                install_hint=(
                    "Place the checkpoint in the repo-level models/ directory."
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
            "Returns L2-normalized 512-D embeddings.",
            "Rendered images and reference images must be RGB HWC tensors in [0, 1].",
        ),
    ),
    "ir101": VerificationModelSpec(
        name="adaface",
        variant="ir101",
        display_name="AdaFace IR-101",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x / 255 - 0.5) / 0.5",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=0.1840,
        requirements=("insightface", "onnxruntime", "torchvision"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "adaface_ir101_ms1mv3.ckpt",
                description="AdaFace IR-101 checkpoint",
                source_url=(
                    "https://drive.google.com/file/d/"
                    "1hRI8YhlfTx2YMzyDwsqLTOxbyFVOqpSI/view?usp=sharing"
                ),
                install_hint=(
                    "Place the checkpoint in the repo-level models/ directory."
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
            "Returns L2-normalized 512-D embeddings.",
            "Rendered images and reference images must be RGB HWC tensors in [0, 1].",
        ),
    ),
}


class Flatten(nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return input.view(input.size(0), -1)


class BasicBlockIR(nn.Module):
    """BasicBlock for IRNet."""

    def __init__(self, in_channel: int, depth: int, stride: int):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res_layer(x) + self.shortcut_layer(x)


class Bottleneck(namedtuple("Block", ["in_channel", "depth", "stride"])):
    """A named tuple describing a ResNet block."""


def get_block(in_channel: int, depth: int, num_units: int, stride: int = 2) -> list[Bottleneck]:
    return [Bottleneck(in_channel, depth, stride)] + [
        Bottleneck(depth, depth, 1) for _ in range(num_units - 1)
    ]


def get_blocks(num_layers: int = 100) -> list[list[Bottleneck]]:
    if num_layers == 50:
        return [
            get_block(64, 64, 3),
            get_block(64, 128, 4),
            get_block(128, 256, 14),
            get_block(256, 512, 3),
        ]
    if num_layers == 100:
        return [
            get_block(in_channel=64, depth=64, num_units=3),
            get_block(in_channel=64, depth=128, num_units=13),
            get_block(in_channel=128, depth=256, num_units=30),
            get_block(in_channel=256, depth=512, num_units=3),
        ]
    raise ValueError(f"num_layers should be 50 or 100, but got {num_layers}")


def initialize_weights(modules) -> None:
    for module in modules:
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.weight is not None:
                module.weight.data.fill_(1)
            if module.bias is not None:
                module.bias.data.zero_()


class Backbone(nn.Module):
    def __init__(self, input_size: tuple[int, int], num_layers: int, mode: str = "ir"):
        super().__init__()
        assert input_size[0] in [112, 224], "input_size should be [112, 112] or [224, 224]"
        assert num_layers in [18, 34, 50, 100, 152, 200], "num_layers should be 18, 34, 50, 100 or 152"
        assert mode in ["ir", "ir_se"], "mode should be ir or ir_se"

        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        blocks = get_blocks(num_layers)
        self.output_layer = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.Dropout(0.4),
            Flatten(),
            nn.Linear(512 * 7 * 7, 512),
            nn.BatchNorm1d(512, affine=False),
        )
        modules = []
        for block in blocks:
            for bottleneck in block:
                modules.append(
                    BasicBlockIR(
                        bottleneck.in_channel,
                        bottleneck.depth,
                        bottleneck.stride,
                    )
                )
        self.body = nn.Sequential(*modules)
        initialize_weights(self.modules())

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        norm = torch.norm(x, 2, 1, True)
        return torch.div(x, norm), norm


def IR_101(input_size: tuple[int, int] = (112, 112)) -> Backbone:
    return Backbone(input_size, 100, "ir")


def IR_50(input_size: tuple[int, int] = (112, 112)) -> Backbone:
    return Backbone(input_size, 50, "ir")


def get_adaface_specs() -> dict[str, VerificationModelSpec]:
    """Return the supported AdaFace variants."""

    return dict(_ADAFACE_SPECS)


def load_pretrained_model(
    path: Path,
    model_type: Literal["ir50", "ir101"] = "ir101",
    device: str = "cuda",
) -> nn.Module:
    """Load a pretrained AdaFace model."""

    spec = _ADAFACE_SPECS[model_type]
    assets = {asset.key: asset for asset in spec.assets}
    checkpoint_path = ensure_asset_present(assets["checkpoint"])

    if model_type == "ir50":
        model = IR_50()
    elif model_type == "ir101":
        model = IR_101()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    state = torch.load(
        checkpoint_path,
        weights_only=False,
        map_location=torch.device(device),
    )
    try:
        state_dict = state["state_dict"]
    except KeyError as error:
        raise ModelAssetMissingError(
            f"AdaFace checkpoint is missing the expected 'state_dict' key: {checkpoint_path}."
        ) from error

    model_state_dict = {
        key[6:]: value for key, value in state_dict.items() if key.startswith("model.")
    }
    model.load_state_dict(model_state_dict)
    model.eval()
    return model


class AdaFaceTorchModel(nn.Module):
    def __init__(
        self,
        path: Path,
        freeze: bool = True,
        model_type: Literal["ir50", "ir101"] = "ir101",
        device: str = "cuda",
    ):
        super().__init__()
        self.model = load_pretrained_model(path, model_type, device)
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            self.model.eval()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(images):
            raise ValueError("Input must be a PyTorch tensor.")
        embeddings, _ = self.model(images)
        return embeddings

    def preprocess_face(self, face_crop: torch.Tensor) -> torch.Tensor:
        face = face_crop.float() / 255.0
        return transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])(face)


class AdaFace(FaceEmbedder):
    """AdaFace embedding model for differentiable face verification."""

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        batch_size: int,
        model_path: Path,
        model_type: Literal["ir50", "ir101"] = "ir50",
    ) -> None:
        self.spec = _ADAFACE_SPECS[model_type]
        self.requested_device = device
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.batch_size = batch_size
        self.ctx_id = 0 if self.device_str == "cuda" else -1
        self.model_path = Path(model_path)

        ort.set_default_logger_severity(3)
        self.validate_assets()
        self.detect_model: FaceDetector = get_shared_insightface_detector(self.ctx_id)
        self.model = AdaFaceTorchModel(
            path=self.model_path,
            model_type=model_type,
            device=self.device_str,
        ).to(self.device)
        self.model.eval()

    def validate_assets(self) -> None:
        assets = {asset.key: asset for asset in self.spec.assets}
        ensure_asset_present(assets["checkpoint"])

    def _align_from_detection(
        self,
        image: torch.Tensor,
        detection: FaceDetection,
    ) -> torch.Tensor:
        M = estimate_norm(detection.landmarks, image_size=self.spec.input_size[0])
        return warp_affine_pytorch(
            image_tensor=image.permute(2, 0, 1).unsqueeze(0).float(),
            m_matrix=torch.from_numpy(M).unsqueeze(0).to(image.device).float(),
            out_size=self.spec.input_size,
        ).squeeze(0)

    def _align_face(self, image: torch.Tensor) -> torch.Tensor:
        detection = detect_face_for_embed(image, self.detect_model)
        if detection is None:
            raise FaceNotDetectedError("No face detected in the image for AdaFace embedding.")
        return self._align_from_detection(image, detection)

    def _prepare_tensor(self, aligned_image: torch.Tensor) -> torch.Tensor:
        tensor = aligned_image.float() * 255.0
        tensor = self.model.preprocess_face(tensor).unsqueeze(0)
        return tensor.to(self.device)

    @staticmethod
    def _norm(vec: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(vec, p=2, dim=1, keepdim=True)
        if torch.any(norm == 0):
            return vec
        return vec / norm

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        aligned = self._align_face(image)
        tensor = self._prepare_tensor(aligned)
        emb = self.model(tensor)
        return self._norm(emb)

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

        batch = torch.cat([self._prepare_tensor(aligned[i]) for i in valid_indices])
        embeddings = self._norm(self.model(batch))  # (N, 512)

        results: list[torch.Tensor | None] = [None] * len(images)
        for out_idx, src_idx in enumerate(valid_indices):
            results[src_idx] = embeddings[out_idx : out_idx + 1]
        return results

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.embed(image)
