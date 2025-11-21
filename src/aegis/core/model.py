import sys
from ..config import ROOT_DIR, DATASETS_DIR

sys.path.append((ROOT_DIR.resolve() / "GaussianAvatars").as_posix())

from .helpers import (
    ensure_output_structure,
    get_foolbox_attack,
    get_targeted_features,
    normalize_camera_angles,
    set_targeted_features,
    write_ply_with_dc_colors,
)
from ..models import get_verification_model
from ..splat import PipelineConfig, load_gaussians, render_single_frame
from ..utils import load_image_from_file, seed_experiment
from gaussian_renderer import FlameGaussianModel
from .pipeline import FaceRenderVerification
from utils.viewer_utils import OrbitCamera
import foolbox as fb
import numpy as np
import torch
from PIL import Image
import shutil
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Tuple


class AEGIS:
    def __init__(
        self,
        embedder_name: str,
        avatar_dir: str | Path,
        target_image: str | Path,
        epsilons: list[float],
        selected_regions: Optional[List[str]] = None,
        camera_boundary_angles: Optional[
            Tuple[float, float, float, float, float, float]
        ] = None,
        angle_aggregation: str = "mean",
        k_angles: int = 5,
        targeted_features: str = "DC",
        adv_attack: str = "linfpgd",
        attack_steps: int = 300,
        ver_threshold: float | None = None,
        seed: int = 42,
        output_name: str = "NeRSembleMasked",
    ):
        # Prepare experiment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        seed_experiment(self.seed)

        # Prepare rendering camera
        self.camera_angles = normalize_camera_angles(camera_boundary_angles)
        self.angle_aggregation = angle_aggregation
        self.k_angles = k_angles
        self.pipeline, self.root_cam = self._init_orbit_cam()

        # Prepare verifier
        self.ver_threshold = ver_threshold
        self.ver_model = get_verification_model(embedder_name, device=self.device)

        # Prepare avatar
        self.targeted_features = targeted_features
        self.gaussians: FlameGaussianModel | None = None
        self.ref_emb = self._get_reference_id_embedding(
            image_path=(
                target_image
                if isinstance(target_image, str)
                else target_image.as_posix()
            )
        )

        # Prepare attack
        self.epsilons = epsilons
        self.adv_attack = adv_attack
        self.attack_steps = attack_steps
        self.mask: torch.Tensor | None = None
        self.selected_regions = selected_regions
        self.att_tensor = self._prepare_tensor_for_attack(avatar_dir)
        self.foolbox_model, self.wrapped_module = self.setup_foolbox_attack()

        # Output settings
        self.avatar_dir = avatar_dir
        self.output_base_name = f"{output_name}_{embedder_name}_"

    def _get_reference_id_embedding(self, image_path: str) -> torch.Tensor:
        ref_image = load_image_from_file(
            image_path
        )  # torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0]
        ref_image = ref_image.permute(1, 2, 0).to(self.device)  # (H, W, C)
        with torch.no_grad():
            ref_embedding = self.ver_model(ref_image)
        return ref_embedding

    def _init_orbit_cam(
        self,
        height: int = 540,
        width: int = 960,
        radius: float = 1,
        fovy: float = 20,
        bg_color: list[float] | None = None,
    ) -> tuple[PipelineConfig, OrbitCamera]:
        if bg_color is None:
            bg_color = [1.0, 1.0, 1.0]
        pipeline = PipelineConfig(background_color=bg_color)
        root_cam = OrbitCamera(width, height, r=radius, fovy=fovy, convention="opencv")
        return pipeline, root_cam

    def _prepare_tensor_for_attack(self, avatar_dir: str | Path) -> torch.Tensor:
        loaded = load_gaussians(
            point_path=Path(avatar_dir) / "point_cloud.ply",
            select_regions=self.selected_regions.copy(),
        )
        if self.selected_regions:
            gaussians, mask = loaded
        else:
            gaussians = loaded
            mask = None

        features = get_targeted_features(gaussians, self.targeted_features)
        if mask is None:
            mask = torch.ones(
                features.shape[0],
                dtype=torch.bool,
                device=features.device,
            )
        else:
            mask = mask.to(features.device)

        self.gaussians = gaussians
        self.mask = mask

        with torch.no_grad():
            att_tensor = features.clone().detach()[mask]
        return att_tensor

    def render_frame_in_rgb(
        self,
        new_features: torch.Tensor,
        orbit_cam: Optional[Tuple[float, float, float]] = None,
    ) -> torch.Tensor:
        if self.gaussians is None or self.mask is None:
            raise RuntimeError("Gaussians and mask must be prepared before rendering.")

        gaussians = deepcopy(self.gaussians)

        if orbit_cam is not None:
            orbit_x, orbit_y, orbit_z = orbit_cam
            if orbit_x != 0:
                self.root_cam.orbit_x(orbit_x)
            if orbit_y != 0:
                self.root_cam.orbit_y(orbit_y)
            if orbit_z != 0:
                self.root_cam.orbit_z(orbit_z)

        features = get_targeted_features(gaussians, self.targeted_features).clone()
        new_features = new_features.to(features.device)
        features[self.mask] = new_features
        set_targeted_features(gaussians, self.targeted_features, features)

        rgb = render_single_frame(gaussians, self.root_cam, self.pipeline)

        if orbit_cam is not None:
            orbit_x, orbit_y, orbit_z = orbit_cam
            if orbit_z != 0:
                self.root_cam.orbit_z(-orbit_z)
            if orbit_y != 0:
                self.root_cam.orbit_y(-orbit_y)
            if orbit_x != 0:
                self.root_cam.orbit_x(-orbit_x)

        return rgb

    def setup_foolbox_attack(self) -> tuple[fb.PyTorchModel, FaceRenderVerification]:
        verifier_model = FaceRenderVerification(
            embedder=self.ver_model,
            reference_embedding=self.ref_emb,
            ver_threshold=self.ver_threshold,
            camera_boundary_angles=self.camera_angles,
            aggregation_mode=self.angle_aggregation,
            k=self.k_angles,
            render_fn=self.render_frame_in_rgb,
        )
        verifier_model.eval()
        verifier_model.to(self.device)

        foolbox_model = fb.PyTorchModel(
            verifier_model,
            bounds=(self.att_tensor.min().item(), self.att_tensor.max().item()),
        )
        return foolbox_model, verifier_model

    def save_results(self, adv_features: torch.Tensor) -> None:
        if self.selected_regions:
            regions_str = "_".join(sorted(self.selected_regions))
            self.output_base_name += regions_str
        else:
            self.output_base_name += "all"

        avatar_id = (
            Path(self.avatar_dir).name
            if isinstance(self.avatar_dir, str)
            else self.avatar_dir.name
        )

        output_name = DATASETS_DIR / f"seed{self.seed}" / self.output_base_name
        ensure_output_structure(output_name, self.epsilons, avatar_id)

        for eps, features in zip(self.epsilons, adv_features):
            with torch.no_grad():
                adv_rgb = self.render_frame_in_rgb(features)

            render_path = (
                output_name / f"eps_{eps:.3f}" / "renders" / f"{avatar_id}.png"
            )
            adv_img_np = (np.clip(adv_rgb.cpu().detach().numpy(), 0, 1) * 255).astype(
                np.uint8
            )
            Image.fromarray(adv_img_np).save(render_path)

            if self.targeted_features == "DC":
                if self.gaussians is None or self.mask is None:
                    raise RuntimeError(
                        "Gaussians and mask must be available to save PLY."
                    )
                orig_ply_path = Path(self.avatar_dir) / "point_cloud.ply"
                orig_flame_path = Path(self.avatar_dir) / "flame_param.npz"
                ply_output_path = (
                    output_name
                    / f"eps_{eps:.3f}"
                    / "avatars"
                    / avatar_id
                    / "point_cloud.ply"
                )
                new_features = get_targeted_features(
                    self.gaussians, self.targeted_features
                ).clone()
                new_features[self.mask] = features.to(new_features.device)
                write_ply_with_dc_colors(
                    original_ply_path=orig_ply_path,
                    new_colors=new_features,
                    output_ply_path=ply_output_path,
                )
                shutil.copyfile(
                    orig_flame_path, ply_output_path.parent / "flame_param.npz"
                )

    def run(self) -> None:
        # with torch.no_grad():
        #     ref_sim = self.wrapped_module.compute_similarity(
        #         self.att_tensor.to(self.device)
        #     ).item()
        # print(f"Initial Aggregated Cosine Similarity: {ref_sim:.4f}")

        original_class = torch.tensor([0], device=self.device)
        target_class = torch.tensor([1], device=self.device)

        attack = get_foolbox_attack(
            adv_attack_name=self.adv_attack,
            steps=self.attack_steps,
            random_start=False,
        )

        epsilons = None if self.adv_attack.lower() == "ddn" else self.epsilons
        # print(f"Starting masking avatar...")
        raw_adv, clipped_adv, is_adv = attack(
            model=self.foolbox_model,
            inputs=self.att_tensor.unsqueeze(0),
            criterion=target_class,
            epsilons=epsilons,
        )

        # if self.adv_attack.lower() == "ddn":
        #     original = self.att_tensor.unsqueeze(0).to(self.device)
        #     final_eps = torch.norm(clipped_adv[0] - original)
        #     print(f"DDN final epsilon used: {final_eps.item():.4f}")

        # print("Masking completed.")
        self.save_results(clipped_adv)

        for eps, adv_features in zip(self.epsilons, clipped_adv):
            with torch.no_grad():
                adv_sim = self.wrapped_module.compute_similarity(
                    adv_features.to(self.device)
                ).item()
            print(f"Epsilon: {eps:.3f} -> Aggregated Cosine Similarity: {adv_sim:.4f}")
