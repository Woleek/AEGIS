"""AEGIS-native face-recognition surrogate ensemble helpers."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch


def parse_model_key(model_key: str) -> Tuple[str, Optional[str]]:
    """Parse a surrogate model key in the form ``model`` or ``model:variant``."""

    normalized = model_key.strip()
    if not normalized:
        raise ValueError("Surrogate model key cannot be empty.")
    if ":" not in normalized:
        return normalized, None

    model_name, variant = normalized.split(":", 1)
    model_name = model_name.strip()
    variant = variant.strip()
    if not model_name:
        raise ValueError(f"Invalid surrogate key '{model_key}': missing model name.")
    if not variant:
        raise ValueError(f"Invalid surrogate key '{model_key}': missing variant.")
    return model_name, variant


def validate_surrogate_keys(surrogate_keys: Sequence[str]) -> None:
    """Validate surrogate count policy and key format."""

    if len(surrogate_keys) < 1:
        raise ValueError("At least 1 surrogate key is required for ensemble mode.")
    for key in surrogate_keys:
        parse_model_key(key)


def precompute_reference_embeddings(
    models: Sequence[object],
    source_image_hwc: torch.Tensor,
) -> List[torch.Tensor]:
    """Precompute detached source (target identity) embeddings per surrogate model.

    Each model maps an HWC image tensor to a ``(1, D)`` embedding. Returns one
    detached reference embedding per model, in the same order as ``models``.
    """

    refs: List[torch.Tensor] = []
    with torch.no_grad():
        for model in models:
            if not callable(model):
                raise TypeError("Surrogate model is not callable.")
            embedding = model(source_image_hwc)
            refs.append(embedding.detach().clone())
    return refs


__all__ = [
    "parse_model_key",
    "validate_surrogate_keys",
    "precompute_reference_embeddings",
]
