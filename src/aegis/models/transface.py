"""TransFace face recognition model implementation for adversarial attacks.

Architecture: VisionTransformer variants.
Reference:    https://github.com/DanJun6737/TransFace
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from insightface.utils.face_align import estimate_norm
from timm.layers import DropPath, trunc_normal_ # type: ignore[import]
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


def _to_2tuple_int(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, tuple):
        if len(value) != 2:
            raise ValueError(f"Expected a 2-tuple, got {value}")
        return int(value[0]), int(value[1])
    return int(value), int(value)


@dataclass(frozen=True)
class _TransFaceVariantConfig:
    embed_dim: int
    depth: int
    num_heads: int
    drop_path_rate: float
    mask_ratio: float


_TRANSFACE_VARIANT_CONFIGS: dict[str, _TransFaceVariantConfig] = {
    "s": _TransFaceVariantConfig(
        embed_dim=512,
        depth=12,
        num_heads=8,
        drop_path_rate=0.05,
        mask_ratio=0.0,
    ),
    "b": _TransFaceVariantConfig(
        embed_dim=512,
        depth=24,
        num_heads=8,
        drop_path_rate=0.05,
        mask_ratio=0.05,
    ),
    "l": _TransFaceVariantConfig(
        embed_dim=768,
        depth=24,
        num_heads=8,
        drop_path_rate=0.05,
        mask_ratio=0.05,
    ),
}


_TRANSFACE_SPECS: dict[str, VerificationModelSpec] = {
    "s": VerificationModelSpec(
        name="transface",
        variant="s",
        display_name="TransFace-S (ViT-S)",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x / 255 - 0.5) / 0.5  [RGB, float32]",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=0.251783,
        requirements=("insightface", "timm"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "transface-s.pt",
                description="TransFace-S checkpoint",
                source_url="https://drive.google.com/file/d/1UZWCg7jNESDv8EWs7mxQSswCMGbAZNF4/view?usp=share_link",
                alt_paths=(
                    MODELS_DIR / "transface_vit_s_dp005_mask_0.pt",
                    MODELS_DIR / "glint360k_model_TransFace_S.pt",
                    MODELS_DIR / "ms1mv2_model_TransFace_S.pt",
                    MODELS_DIR / "transface" / "transface_s_ms1mv2.pt",
                ),
                install_hint=(
                    "Download a TransFace-S checkpoint from the official repo README and "
                    "place it at models/transface-s.pt."
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
            "Input to embed() must be RGB HWC tensor in [0, 1].",
            "Model outputs 512-D features; downstream cosine similarity is used directly.",
            "Threshold is not set; calibrate with --threshold.",
        ),
    ),
    "b": VerificationModelSpec(
        name="transface",
        variant="b",
        display_name="TransFace-B (ViT-B)",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x / 255 - 0.5) / 0.5  [RGB, float32]",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=None,
        requirements=("insightface", "timm"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "transface-b.pt",
                description="TransFace-B checkpoint",
                source_url="https://drive.google.com/file/d/16O-q30mH8d3lECqa5eJd8rABaUlNhQ0K/view?usp=share_link",
                alt_paths=(
                    MODELS_DIR / "transface_vit_b_dp005_mask_005.pt",
                    MODELS_DIR / "glint360k_model_TransFace_B.pt",
                    MODELS_DIR / "ms1mv2_model_TransFace_B.pt",
                ),
                install_hint=(
                    "Download a TransFace-B checkpoint from the official repo README and "
                    "place it at models/transface-b.pt."
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
            "Input to embed() must be RGB HWC tensor in [0, 1].",
            "Model outputs 512-D features; downstream cosine similarity is used directly.",
            "Threshold is not set; calibrate with --threshold.",
        ),
    ),
    "l": VerificationModelSpec(
        name="transface",
        variant="l",
        display_name="TransFace-L (ViT-L)",
        embedding_dim=512,
        input_size=(112, 112),
        normalization="(x / 255 - 0.5) / 0.5  [RGB, float32]",
        detector="InsightFace buffalo_l / det_10g.onnx",
        alignment="5-point similarity transform via estimate_norm(image_size=112)",
        threshold=None,
        requirements=("insightface", "timm"),
        assets=(
            ModelAssetSpec(
                key="checkpoint",
                path=MODELS_DIR / "transface-l.pt",
                description="TransFace-L checkpoint (official architecture)",
                source_url="https://drive.google.com/file/d/1uXUFT6ujEPqvCTHzONsp6-DMIc24Cc85/view?usp=share_link",
                alt_paths=(
                    MODELS_DIR / "transface_vit_l_dp005_mask_005.pt",
                    MODELS_DIR / "glint360k_model_TransFace_L.pt",
                    MODELS_DIR / "ms1mv2_model_TransFace_L.pt",
                ),
                install_hint=(
                    "Download a TransFace-L checkpoint from the official repo README and "
                    "place it at models/transface-l.pt."
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
            "Input to embed() must be RGB HWC tensor in [0, 1].",
            "Model outputs 512-D features; downstream cosine similarity is used directly.",
            "Threshold is not set; calibrate with --threshold.",
        ),
    ),
}


class _Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int, drop: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.ReLU6()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool, attn_drop: float, proj_drop: float):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_token, embed_dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size,
            num_token,
            3,
            self.num_heads,
            embed_dim // self.num_heads,
        ).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(batch_size, num_token, embed_dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class _Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float,
        qkv_bias: bool,
        drop: float,
        attn_drop: float,
        drop_path: float,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = _Attention(
            dim=dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = _Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, drop=drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _PatchEmbed(nn.Module):
    def __init__(
        self,
        img_size: int | tuple[int, int],
        patch_size: int | tuple[int, int],
        in_channels: int,
        embed_dim: int,
    ):
        super().__init__()
        img_size = _to_2tuple_int(img_size)
        patch_size = _to_2tuple_int(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, height, width = x.shape
        if height != self.img_size[0] or width != self.img_size[1]:
            raise ModelAssetMissingError(
                f"TransFace expects input size {self.img_size}, got {(height, width)}."
            )
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class _VisionTransformer(nn.Module):
    def __init__(
        self,
        *,
        img_size: int,
        patch_size: int,
        num_classes: int,
        embed_dim: int,
        depth: int,
        num_heads: int,
        drop_rate: float,
        attn_drop_rate: float,
        drop_path_rate: float,
        mask_ratio: float,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_ratio = mask_ratio
        self.patch_embed = _PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=3,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        drop_path_values = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                _Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path_values[i],
                )
                for i in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(embed_dim)

        self.senet = nn.Sequential(
            nn.Linear(in_features=embed_dim * self.num_patches, out_features=self.num_patches, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=self.num_patches, out_features=self.num_patches, bias=False),
            nn.Sigmoid(),
        )

        self.feature = nn.Sequential(
            nn.Linear(in_features=embed_dim * self.num_patches, out_features=embed_dim, bias=False),
            nn.BatchNorm1d(num_features=embed_dim, eps=2e-5),
            nn.Linear(in_features=embed_dim, out_features=num_classes, bias=False),
            nn.BatchNorm1d(num_features=num_classes, eps=2e-5),
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.normal_(self.mask_token, std=0.02)
        trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x.float())
        original = x
        out = torch.reshape(x, (batch_size, self.num_patches * self.embed_dim))
        out = self.senet(out)
        out_softmax = out.softmax(dim=1)
        out = torch.reshape(out, (batch_size, self.num_patches, 1))
        out = out * original
        out = torch.reshape(out, (batch_size, self.num_patches * self.embed_dim))
        return out, out_softmax

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, weight = self.forward_features(x)
        out_x = torch.reshape(x, (x.shape[0], self.num_patches, self.embed_dim))
        patch_std = torch.std(out_x, dim=2)
        x = self.feature(x)
        return x, weight, patch_std


def _build_transface_backbone(variant: Literal["s", "b", "l"]) -> _VisionTransformer:
    config = _TRANSFACE_VARIANT_CONFIGS[variant]
    return _VisionTransformer(
        img_size=112,
        patch_size=9,
        num_classes=512,
        embed_dim=config.embed_dim,
        depth=config.depth,
        num_heads=config.num_heads,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=config.drop_path_rate,
        mask_ratio=config.mask_ratio,
    )


def _extract_state_dict(raw_state: object) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state, dict):
        raise ModelAssetMissingError("TransFace checkpoint is not a state dict.")

    for key in ("state_dict", "model", "backbone", "net"):
        candidate = raw_state.get(key)
        if isinstance(candidate, dict) and candidate:
            raw_state = candidate
            break

    if not isinstance(raw_state, dict):
        raise ModelAssetMissingError("TransFace checkpoint payload is malformed.")

    state_dict = {
        key: value
        for key, value in raw_state.items()
        if isinstance(value, torch.Tensor)
    }
    if not state_dict:
        raise ModelAssetMissingError("TransFace checkpoint does not contain tensor weights.")

    normalized_state_dict: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        normalized_key = key
        for prefix in ("module.", "backbone.", "model."):
            if normalized_key.startswith(prefix):
                normalized_key = normalized_key[len(prefix):]
        normalized_state_dict[normalized_key] = value

    return normalized_state_dict


def get_transface_specs() -> dict[str, VerificationModelSpec]:
    return dict(_TRANSFACE_SPECS)


class TransFace:
    spec: VerificationModelSpec
    device: torch.device

    def __init__(
        self,
        device: Literal["cpu", "cuda"],
        model_path: Path,
        variant: Literal["s", "b", "l"] = "s",
    ) -> None:
        self.spec = _TRANSFACE_SPECS[variant]
        self.device_str = resolve_compute_device(device)
        self.device = torch.device(self.device_str)
        self.ctx_id = 0 if self.device_str == "cuda" else -1

        self.validate_assets()
        self.detect_model: FaceDetector = get_shared_insightface_detector(self.ctx_id)

        assets = {asset.key: asset for asset in self.spec.assets}
        checkpoint_path = ensure_asset_present(assets["checkpoint"])
        raw_state = torch.load(checkpoint_path, map_location=self.device)
        state_dict = _extract_state_dict(raw_state)

        model = _build_transface_backbone(variant)
        model_state = model.state_dict()
        filtered_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing_keys, unexpected_keys = model.load_state_dict(filtered_state, strict=False)
        if missing_keys or unexpected_keys:
            raise ModelAssetMissingError(
                "TransFace checkpoint is incompatible with the selected variant. "
                f"Missing keys: {missing_keys[:5]} | Unexpected keys: {unexpected_keys[:5]}"
            )
        self.model = model.eval().to(self.device)

    def validate_assets(self) -> None:
        assets = {asset.key: asset for asset in self.spec.assets}
        ensure_asset_present(assets["checkpoint"])

    def _align_from_detection(
        self,
        image: torch.Tensor,
        detection: FaceDetection,
    ) -> torch.Tensor:
        matrix = estimate_norm(detection.landmarks, image_size=self.spec.input_size[0])
        aligned = warp_affine_pytorch(
            image_tensor=image.permute(2, 0, 1).unsqueeze(0).float(),
            m_matrix=torch.from_numpy(matrix).unsqueeze(0).to(image.device).float(),
            out_size=self.spec.input_size,
        )
        return aligned.squeeze(0)  # (C, H, W)

    def _align_face(self, image: torch.Tensor) -> torch.Tensor:
        detection = detect_face_for_embed(image, self.detect_model)
        if detection is None:
            raise FaceNotDetectedError("No face detected in the image for TransFace embedding.")
        return self._align_from_detection(image, detection)

    def _normalize(self, aligned: torch.Tensor) -> torch.Tensor:
        return aligned.sub(0.5).div(0.5).unsqueeze(0).to(self.device)

    def embed(self, image: torch.Tensor) -> torch.Tensor:
        aligned = self._align_face(image)
        tensor = self._normalize(aligned)
        features, _, _ = self.model(tensor)
        norm = torch.norm(features, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        return features / norm

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
        features, _, _ = self.model(batch)
        norm = torch.norm(features, p=2, dim=1, keepdim=True).clamp(min=1e-12)
        embeddings = features / norm  # (N, 512)

        results: list[torch.Tensor | None] = [None] * len(images)
        for out_idx, src_idx in enumerate(valid_indices):
            results[src_idx] = embeddings[out_idx : out_idx + 1]
        return results

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return self.embed(image)
