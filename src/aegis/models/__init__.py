import torch
from ..config import ROOT_DIR, MODELS_DIR
from .adaface import AdaFaceEmbedder, AdaFace
from .arcface import ArcFaceEmbedder, ArcFace
from .swinface_embedder import SwinFaceEmbedder, SwinFace
from .transface_embedder import TransFaceEmbedder, TransFace
from .facenet_embedder import FaceNetEmbedder, FaceNet
from .cosface_embedder import CosFaceEmbedder, CosFace
from .base import BaseEmbedder, resolve_compute_device


def get_verification_model(
    model_name: str, device: torch.device
) -> ArcFace | AdaFace | SwinFace | TransFace | FaceNet | CosFace:
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
    elif model_name == "swinface":
        embedder = SwinFace(
            device=device_str,
            batch_size=1,
            model_path=MODELS_DIR / "swinface" / "checkpoint_step_79999_gpu_0.pt",
        )
    elif model_name == "transface":
        embedder = TransFace(
            device=device_str,
            batch_size=1,
            model_path=MODELS_DIR / "transface" / "transface_s_ms1mv2.pt",
            model_type="vit_s",
        )
    elif model_name == "facenet":
        embedder = FaceNet(
            device=device_str,
            batch_size=1,
            model_path=MODELS_DIR / "facenet" / "facenet_vggface2.pth",
        )
    elif model_name == "cosface":
        embedder = CosFace(
            device=device_str,
            batch_size=1,
            model_path=MODELS_DIR / "cosface" / "cosface_ir50_ms1mv2.pth",
        )
    else:
        raise ValueError(f"Unsupported embedder model: {model_name}")
    return embedder


__all__ = [
    "AdaFaceEmbedder",
    "ArcFaceEmbedder",
    "AdaFace",
    "ArcFace",
    "SwinFaceEmbedder",
    "SwinFace",
    "TransFaceEmbedder",
    "TransFace",
    "FaceNetEmbedder",
    "FaceNet",
    "CosFaceEmbedder",
    "CosFace",
    "BaseEmbedder",
    "resolve_compute_device",
    "get_verification_model",
]
