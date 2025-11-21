from ..models import AdaFace, ArcFace
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
        embedder: ArcFace | AdaFace,
        reference_embedding: torch.Tensor,
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
    ):
        super().__init__()
        self.model = embedder
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
        # Register the reference embedding as a buffer
        self.register_buffer("ref_emb", reference_embedding)

    def forward(self, new_features: torch.Tensor) -> torch.Tensor:
        similarity = self._compute_aggregated_similarity(new_features)
        logits = self._similarity_to_logits(similarity)
        return logits

    def compute_similarity(self, new_features: torch.Tensor) -> torch.Tensor:
        sim = self._compute_aggregated_similarity(new_features)
        return sim

    def _compute_aggregated_similarity(self, features: torch.Tensor) -> torch.Tensor:
        per_view_similarities: list[torch.Tensor] = []
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
        for orbit in sample_angles:
            att_rgb = self.render_fn(features, orbit_cam=orbit)
            try:
                att_emb = self.model(att_rgb)
            except:
                Image.fromarray(
                    (att_rgb.detach().cpu().numpy() * 255).astype(np.uint8)
                ).save("error_input.png")
                input("Check input image for cam {}...".format(orbit))
            similarity = torch.cosine_similarity(
                att_emb,
                self.ref_emb.expand_as(att_emb),
                dim=1,
            )
            per_view_similarities.append(similarity.squeeze(0))

        view_sim_tensor = torch.stack(per_view_similarities, dim=0)
        aggregated_similarity = self._aggregate_similarities(view_sim_tensor).unsqueeze(
            0
        )
        return aggregated_similarity

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
