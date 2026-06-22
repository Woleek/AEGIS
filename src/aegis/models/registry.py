"""Registry for face verification model implementations."""

from __future__ import annotations

from collections.abc import Callable

import torch

from .base import (
    FaceEmbedder,
    UnsupportedModelVariantError,
    VerificationModelSpec,
)

VerificationModelFactory = Callable[[torch.device, str | None], FaceEmbedder]
VerificationModelSpecProvider = Callable[[], dict[str, VerificationModelSpec]]

_MODEL_FACTORIES: dict[str, VerificationModelFactory] = {}
_MODEL_SPEC_PROVIDERS: dict[str, VerificationModelSpecProvider] = {}


def register_verification_model(
    model_name: str,
    factory: VerificationModelFactory,
    spec_provider: VerificationModelSpecProvider,
) -> None:
    """Register a face verification model and its supported variants."""

    _MODEL_FACTORIES[model_name] = factory
    _MODEL_SPEC_PROVIDERS[model_name] = spec_provider


def get_verification_model(
    model_name: str,
    device: torch.device | str,
    variant: str | None = None,
) -> FaceEmbedder:
    """Instantiate a registered verification model."""

    try:
        factory = _MODEL_FACTORIES[model_name]
    except KeyError as error:
        available = ", ".join(sorted(_MODEL_FACTORIES)) or "<none>"
        raise ValueError(
            f"Unsupported embedder model: {model_name}. Available models: {available}."
        ) from error
    resolved_device = device if isinstance(device, torch.device) else torch.device(device)
    return factory(resolved_device, variant)


def get_model_specs(model_name: str | None = None) -> dict[str, VerificationModelSpec]:
    """Return registered model specs.

    When `model_name` is provided, keys are variant names.
    When omitted, keys are fully qualified `model:variant` ids.
    """

    if model_name is not None:
        try:
            return dict(_MODEL_SPEC_PROVIDERS[model_name]())
        except KeyError as error:
            raise ValueError(f"Unknown verification model: {model_name}") from error

    specs: dict[str, VerificationModelSpec] = {}
    for provider in _MODEL_SPEC_PROVIDERS.values():
        for spec in provider().values():
            specs[spec.model_id] = spec
    return specs


def get_model_spec(model_name: str, variant: str | None = None) -> VerificationModelSpec:
    """Return the concrete spec for a model and variant."""

    specs = get_model_specs(model_name)
    if not specs:
        raise ValueError(f"Model '{model_name}' did not register any variants.")
    resolved_variant = variant or next(iter(specs))
    try:
        return specs[resolved_variant]
    except KeyError as error:
        available = ", ".join(sorted(specs))
        raise UnsupportedModelVariantError(
            f"Unsupported variant '{resolved_variant}' for model '{model_name}'. "
            f"Available variants: {available}."
        ) from error


def list_verification_models() -> dict[str, tuple[str, ...]]:
    """Return registered model names and their supported variants."""

    return {
        model_name: tuple(get_model_specs(model_name).keys())
        for model_name in sorted(_MODEL_FACTORIES)
    }
