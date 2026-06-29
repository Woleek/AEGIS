from ..models import AdaFace, ArcFace
from ..models.base import FaceNotDetectedError, detect_faces_for_embed
import numpy as np
import torch
from PIL import Image
import random
from typing import Callable, Tuple, Optional


class FaceRenderVerification(torch.nn.Module):
    """
    A wrapper that makes the ArcFace / AdaFace verification system look like a classifier for foolbox.
    """

    def __init__(
        self,
        embedder: ArcFace | AdaFace | None = None,
        reference_embedding: torch.Tensor | None = None,
        ver_threshold: float | None = None,
        camera_boundary_angles: Optional[
            list[Tuple[float, float, float, float, float, float]]
        ] = None,
        aggregation_mode: str = "mean",
        k: int = 5,
        render_fn: (
            Callable[[torch.Tensor, Optional[Tuple[float, float, float]]], torch.Tensor]
            | None
        ) = None,
        embedders: Optional[list] = None,
        reference_embeddings: Optional[list[torch.Tensor]] = None,
        model_weights: Optional[list[float]] = None,
        cross_model_aggregation: str = "mean",
    ):
        super().__init__()
        # --- Ensemble vs single-model resolution -------------------------------
        # Default / backward-compatible path: a single `embedder` + single
        # `reference_embedding`. Ensemble path is opt-in via `embedders` +
        # `reference_embeddings` (a per-model reference embedding each).
        if embedders is not None:
            if reference_embeddings is None:
                raise ValueError(
                    "reference_embeddings must be provided when embedders is given."
                )
            if len(embedders) != len(reference_embeddings):
                raise ValueError(
                    "embedders and reference_embeddings must have the same length."
                )
            if len(embedders) == 0:
                raise ValueError("embedders must contain at least one model.")
            self.ensemble = True
            # Keep models in a ModuleList where possible so .to(device)/.eval()
            # propagate; fall back to a plain list for non-Module callables.
            if all(isinstance(m, torch.nn.Module) for m in embedders):
                self.models = torch.nn.ModuleList(embedders)
            else:
                self.models = list(embedders)
            # Single-model attribute kept pointing at the first model for any
            # introspection code; the single-model forward path is not used here.
            self.model = embedders[0]
        else:
            if embedder is None:
                raise ValueError(
                    "Either embedder (single-model) or embedders (ensemble) "
                    "must be provided."
                )
            if reference_embedding is None:
                raise ValueError(
                    "reference_embedding must be provided in single-model mode."
                )
            self.ensemble = False
            self.model = embedder
        self.cross_model_aggregation = cross_model_aggregation.lower()
        self.model_weights = model_weights
        self.ver_threshold = ver_threshold
        self.aggregation_mode = aggregation_mode.lower()
        if camera_boundary_angles is None or len(camera_boundary_angles) == 0:
            camera_boundary_angles = [(0.0, 0.0, 0.0)]
        self.camera_boundary_angles = [
            (
                float(orbit_x_min),
                float(orbit_x_max),
                float(orbit_y_min),
                float(orbit_y_max),
                float(orbit_z_min),
                float(orbit_z_max),
            )
            for orbit_x_min, orbit_x_max, orbit_y_min, orbit_y_max, orbit_z_min, orbit_z_max in camera_boundary_angles
        ]
        self.k = k
        if render_fn is None:
            raise ValueError("render_fn must be provided for FaceRenderVerification")
        self.render_fn = render_fn
        if self.aggregation_mode not in {"mean", "max", "min", "median"}:
            raise ValueError(
                "Unsupported aggregation mode. Choose from ['mean', 'max', 'min', 'median']."
            )
        if self.ensemble:
            # Register one reference embedding buffer per surrogate model so that
            # .to(device) moves all of them. Buffers are named ref_emb_{i}.
            self._n_models = len(reference_embeddings)
            for i, ref in enumerate(reference_embeddings):
                self.register_buffer(f"ref_emb_{i}", ref)
            # Validate / default cross-model weights.
            if self.model_weights is not None:
                if len(self.model_weights) != self._n_models:
                    raise ValueError(
                        "model_weights length must match the number of embedders."
                    )
            if self.cross_model_aggregation not in {"mean", "max", "min", "median"}:
                raise ValueError(
                    "Unsupported cross_model_aggregation. Choose from "
                    "['mean', 'max', 'min', 'median']."
                )
        else:
            # Register the reference embedding as a buffer (single-model, unchanged)
            self.register_buffer("ref_emb", reference_embedding)

    def forward(self, new_features: torch.Tensor) -> torch.Tensor:
        similarity = self._compute_aggregated_similarity(new_features)
        logits = self._similarity_to_logits(similarity)
        return logits

    def compute_similarity(self, new_features: torch.Tensor) -> torch.Tensor:
        sim = self._compute_aggregated_similarity(new_features)
        return sim

    def _sample_angles(self) -> list[Tuple[float, float, float]]:
        sample_angles: list[Tuple[float, float, float]] = []
        for (
            orbit_x_min,
            orbit_x_max,
            orbit_y_min,
            orbit_y_max,
            orbit_z_min,
            orbit_z_max,
        ) in self.camera_boundary_angles:
            for _ in range(self.k):
                orbit_x = random.uniform(orbit_x_min, orbit_x_max)
                orbit_y = random.uniform(orbit_y_min, orbit_y_max)
                orbit_z = random.uniform(orbit_z_min, orbit_z_max)
                sample_angles.append((orbit_x, orbit_y, orbit_z))
        return sample_angles

    def _render_views(
        self, features: torch.Tensor, sample_angles: list[Tuple[float, float, float]]
    ) -> tuple[list[torch.Tensor], "list"]:
        """Render every sampled view once and detect each face a single time.

        Returns the list of rendered (H, W, C) tensors (gradient-carrying) and the matching list of `FaceDetection | None`.
        """
        rendered = [
            self.render_fn(features, orbit_cam=orbit) for orbit in sample_angles
        ]
        base_model = self.models[0] if self.ensemble else self.model
        detector = getattr(base_model, "detect_model", None)
        if detector is not None:
            detections = detect_faces_for_embed(rendered, detector)
        else:
            detections = [None] * len(rendered)
        return rendered, detections

    def _model_view_similarity(
        self,
        model,
        ref_emb: torch.Tensor,
        rendered: list[torch.Tensor],
        detections: "list",
    ) -> torch.Tensor:
        """One batched recognition forward over all views, then view aggregation.

        Views whose face could not be detected/aligned are skipped. Gradients flow from each embedding back to the rendered images and thus to `features`.
        """
        embeddings = model.embed_batch(rendered, detections=detections)
        per_view: list[torch.Tensor] = []
        for emb in embeddings:
            if emb is None:
                continue
            sim = torch.cosine_similarity(emb, ref_emb.expand_as(emb), dim=1)
            per_view.append(sim.squeeze(0))
        if not per_view:
            raise FaceNotDetectedError(
                "No face detected/aligned in any sampled view for a surrogate model."
            )
        view_sim_tensor = torch.stack(per_view, dim=0)
        return self._aggregate_similarities(view_sim_tensor)

    def _compute_aggregated_similarity(self, features: torch.Tensor) -> torch.Tensor:
        if self.ensemble:
            return self._compute_ensemble_similarity(features)

        sample_angles = self._sample_angles()
        rendered, detections = self._render_views(features, sample_angles)
        aggregated_similarity = self._model_view_similarity(
            self.model, self.ref_emb, rendered, detections
        ).unsqueeze(0)
        return aggregated_similarity

    def _compute_ensemble_similarity(self, features: torch.Tensor) -> torch.Tensor:
        """Ensemble path: per-(model, view) similarity, aggregated over views per
        model (existing aggregation), then aggregated across models into a scalar.

        Each rendered view is shared across all surrogate models so the renderer
        runs once per view, the face is detected once per view, and each model
        does a single batched recognition forward over the views. Gradients
        still flow to `features` through every model. The final scalar goes
        through `_similarity_to_logits` unchanged.
        """
        sample_angles = self._sample_angles()
        rendered, detections = self._render_views(features, sample_angles)

        per_model_sim: list[torch.Tensor] = []
        for i in range(self._n_models):
            ref_emb = getattr(self, f"ref_emb_{i}")
            per_model_sim.append(
                self._model_view_similarity(
                    self.models[i], ref_emb, rendered, detections
                )
            )

        model_sim_tensor = torch.stack(per_model_sim, dim=0)  # (n_models,)
        aggregated = self._aggregate_across_models(model_sim_tensor).unsqueeze(0)
        return aggregated

    def _aggregate_across_models(self, similarities: torch.Tensor) -> torch.Tensor:
        """Aggregate per-model scalar similarities into one scalar.

        Default is an (optionally weighted) mean over models. If `model_weights`
        is provided, a weighted mean is used regardless of `cross_model_aggregation`
        to keep weighting meaningful. Otherwise the chosen aggregation applies.
        """
        if self.model_weights is not None:
            weights = torch.as_tensor(
                self.model_weights,
                dtype=similarities.dtype,
                device=similarities.device,
            )
            weights = weights / weights.sum()
            return (similarities * weights).sum(dim=0)
        if self.cross_model_aggregation == "mean":
            return similarities.mean(dim=0)
        if self.cross_model_aggregation == "max":
            return similarities.max(dim=0).values
        if self.cross_model_aggregation == "min":
            return similarities.min(dim=0).values
        if self.cross_model_aggregation == "median":
            return similarities.median(dim=0).values
        raise RuntimeError("Invalid cross_model_aggregation encountered.")

    def _aggregate_similarities(self, similarities: torch.Tensor) -> torch.Tensor:
        if self.aggregation_mode == "mean":
            return similarities.mean(dim=0)
        if self.aggregation_mode == "max":
            return similarities.max(dim=0).values
        if self.aggregation_mode == "min":
            return similarities.min(dim=0).values
        if self.aggregation_mode == "median":
            return similarities.median(dim=0).values
        raise RuntimeError("Invalid aggregation mode encountered during forward pass.")

    def _similarity_to_logits(self, similarity: torch.Tensor) -> torch.Tensor:
        if self.ver_threshold is None:
            logits = torch.stack((-similarity * 10.0, similarity * 10.0), dim=1)
        else:
            logits = torch.stack(
                (
                    (self.ver_threshold - similarity) * 10.0,
                    (similarity - self.ver_threshold) * 10.0,
                ),
                dim=1,
            )
        return logits
