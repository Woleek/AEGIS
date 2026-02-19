"""TransFace model module.

TransFace: Calibrating Transformer Training for Face Recognition (ICCV 2023)
https://github.com/DanJun6737/TransFace
"""

from .vit import VisionTransformer, get_model, vit_s, vit_b, vit_l, vit_t

__all__ = [
    "VisionTransformer",
    "get_model",
    "vit_s",
    "vit_b",
    "vit_l",
    "vit_t",
]
