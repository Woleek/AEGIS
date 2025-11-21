import torch
from ..config import ROOT_DIR
from .adaface import AdaFaceEmbedder, AdaFace
from .arcface import ArcFaceEmbedder, ArcFace
from .base import BaseEmbedder, resolve_compute_device


def get_verification_model(model_name: str, device: torch.device) -> ArcFace | AdaFace:
    device_str = "cuda" if device.type == "cuda" else "cpu"

    if model_name == "arcface":
        embedder = ArcFace(device=device_str, batch_size=1)
    elif model_name == "adaface":
        embedder = AdaFace(
            device=device_str,
            batch_size=1,
            model_path=ROOT_DIR / "models",
            model_type="ir50",
        )
    else:
        raise ValueError(f"Unsupported embedder model: {model_name}")
    return embedder


__all__ = [
    "AdaFaceEmbedder",
    "ArcFaceEmbedder",
    "AdaFace",
    "ArcFace",
    "BaseEmbedder",
    "resolve_compute_device",
    "get_verification_model",
]
