"""Face verification models for AEGIS (registry-based)."""

from __future__ import annotations

from typing import Literal, cast

import torch

from ..config import MODELS_DIR
from .adaface import AdaFace, get_adaface_specs
from .arcface import ArcFace, get_arcface_specs
from .cosface import CosFace, get_cosface_specs
from .facenet import FaceNet, get_facenet_specs
from .ir152 import IR152, get_ir152_specs
from .irse50 import IRSE50, get_irse50_specs
from .mobileface import MobileFace, get_mobileface_specs
from .swinface import SwinFace, get_swinface_specs
from .transface import TransFace, get_transface_specs
from .base import (
    BaseEmbedder,
    FaceEmbedder,
    FaceNotDetectedError,
    FaceVerificationError,
    ModelAssetMissingError,
    ModelAssetSpec,
    UnsupportedModelVariantError,
    VerificationModelSpec,
    resolve_compute_device,
)
from .registry import (
    get_model_spec,
    get_model_specs,
    get_verification_model,
    list_verification_models,
    register_verification_model,
)
from .eval_adapter import EvalEmbedderAdapter, get_eval_embedder


def _build_adaface(device: torch.device, variant: str | None = None) -> AdaFace:
    resolved_variant = variant or "ir50"
    if resolved_variant not in {"ir50", "ir101"}:
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model 'adaface'. "
            "Available variants: ir50, ir101."
        )
    return AdaFace(
        device="cuda" if device.type == "cuda" else "cpu",
        batch_size=1,
        model_path=MODELS_DIR,
        model_type=cast(Literal["ir50", "ir101"], resolved_variant),
    )


register_verification_model(
    model_name="adaface",
    factory=_build_adaface,
    spec_provider=get_adaface_specs,
)


def _build_arcface(device: torch.device, variant: str | None = None) -> ArcFace:
    resolved_variant = variant or "r50"
    if resolved_variant not in {"r50", "r100"}:
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model 'arcface'. "
            "Available variants: r50, r100."
        )
    return ArcFace(
        device="cuda" if device.type == "cuda" else "cpu",
        model_path=MODELS_DIR,
        variant=cast(Literal["r50", "r100"], resolved_variant),
    )


register_verification_model(
    model_name="arcface",
    factory=_build_arcface,
    spec_provider=get_arcface_specs,
)


def _build_swinface(device: torch.device, variant: str | None = None) -> SwinFace:
    resolved_variant = variant or "swin_t"
    if resolved_variant not in {"swin_t"}:
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model 'swinface'. "
            "Available variants: swin_t."
        )
    return SwinFace(
        device="cuda" if device.type == "cuda" else "cpu",
        model_path=MODELS_DIR,
        variant=cast(Literal["swin_t"], resolved_variant),
    )


register_verification_model(
    model_name="swinface",
    factory=_build_swinface,
    spec_provider=get_swinface_specs,
)


def _build_facenet(device: torch.device, variant: str | None = None) -> FaceNet:
    resolved_variant = variant or "vggface2"
    if resolved_variant not in {"vggface2", "casia-webface"}:
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model 'facenet'. "
            "Available variants: vggface2, casia-webface."
        )
    return FaceNet(
        device="cuda" if device.type == "cuda" else "cpu",
        variant=cast(Literal["vggface2", "casia-webface"], resolved_variant),
    )


register_verification_model(
    model_name="facenet",
    factory=_build_facenet,
    spec_provider=get_facenet_specs,
)


def _build_transface(device: torch.device, variant: str | None = None) -> TransFace:
    resolved_variant = variant or "s"
    if resolved_variant not in {"s", "b", "l"}:
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model 'transface'. "
            "Available variants: s, b, l."
        )
    return TransFace(
        device="cuda" if device.type == "cuda" else "cpu",
        model_path=MODELS_DIR,
        variant=cast(Literal["s", "b", "l"], resolved_variant),
    )


register_verification_model(
    model_name="transface",
    factory=_build_transface,
    spec_provider=get_transface_specs,
)


def _build_ir152(device: torch.device, variant: str | None = None) -> IR152:
    resolved_variant = variant or "r152"
    if resolved_variant not in {"r152"}:
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model 'ir152'. "
            "Available variants: r152."
        )
    return IR152(
        device="cuda" if device.type == "cuda" else "cpu",
        model_path=MODELS_DIR,
        variant="r152",
    )


register_verification_model(
    model_name="ir152",
    factory=_build_ir152,
    spec_provider=get_ir152_specs,
)


def _build_irse50(device: torch.device, variant: str | None = None) -> IRSE50:
    resolved_variant = variant or "ir_se50"
    if resolved_variant not in {"ir_se50"}:
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model 'irse50'. "
            "Available variants: ir_se50."
        )
    return IRSE50(
        device="cuda" if device.type == "cuda" else "cpu",
        model_path=MODELS_DIR,
        variant="ir_se50",
    )


register_verification_model(
    model_name="irse50",
    factory=_build_irse50,
    spec_provider=get_irse50_specs,
)


def _build_mobileface(device: torch.device, variant: str | None = None) -> MobileFace:
    resolved_variant = variant or "default"
    if resolved_variant not in {"default"}:
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model 'mobileface'. "
            "Available variants: default."
        )
    return MobileFace(
        device="cuda" if device.type == "cuda" else "cpu",
        model_path=MODELS_DIR,
        variant="default",
    )


register_verification_model(
    model_name="mobileface",
    factory=_build_mobileface,
    spec_provider=get_mobileface_specs,
)


def _build_cosface(device: torch.device, variant: str | None = None) -> CosFace:
    resolved_variant = variant or "ir_se50"
    if resolved_variant not in {"ir_se50"}:
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model 'cosface'. "
            "Available variants: ir_se50."
        )
    return CosFace(
        device="cuda" if device.type == "cuda" else "cpu",
        model_path=MODELS_DIR,
        variant="ir_se50",
    )


register_verification_model(
    model_name="cosface",
    factory=_build_cosface,
    spec_provider=get_cosface_specs,
)


__all__ = [
    "AdaFace",
    "ArcFace",
    "CosFace",
    "FaceNet",
    "IR152",
    "IRSE50",
    "MobileFace",
    "SwinFace",
    "TransFace",
    "BaseEmbedder",
    "EvalEmbedderAdapter",
    "FaceEmbedder",
    "FaceNotDetectedError",
    "FaceVerificationError",
    "ModelAssetMissingError",
    "ModelAssetSpec",
    "UnsupportedModelVariantError",
    "VerificationModelSpec",
    "resolve_compute_device",
    "get_eval_embedder",
    "get_model_spec",
    "get_model_specs",
    "get_verification_model",
    "list_verification_models",
]
