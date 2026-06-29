import sys
from ..config import ROOT_DIR, DATASETS_DIR

sys.path.append((ROOT_DIR.resolve() / "GaussianAvatars").as_posix())

from .helpers import (
    adaptive_linf_pgd_attack,
    create_adaptive_epsilon_mask,
    ensure_output_structure,
    DEFAULT_REGION_MULTIPLIERS,
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
from .fr_ensemble import (
    parse_model_key,
    precompute_reference_embeddings,
    validate_surrogate_keys,
)
from utils.viewer_utils import OrbitCamera
import foolbox as fb
import numpy as np
import torch
from PIL import Image
import shutil
from pathlib import Path
from typing import List, Optional, Tuple
from tqdm import tqdm


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
        adaptive_epsilon: bool = False,
        region_multipliers: Optional[dict[str, float]] = None,
        radius: float = 1,
        surrogate_keys: Optional[List[str]] = None,
        cross_model_aggregation: str = "mean",
        model_weights: Optional[List[float]] = None,
        checkpoint_steps: Optional[List[int]] = None,
        eval_viewpoints: Optional[List[Tuple[float, float, float]]] = None,
    ):
        # Prepare experiment
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed
        seed_experiment(self.seed)

        # Prepare rendering camera
        self.camera_angles = normalize_camera_angles(camera_boundary_angles)
        self.angle_aggregation = angle_aggregation
        self.k_angles = k_angles
        self.pipeline, self.root_cam = self._init_orbit_cam(radius=radius)

        # Prepare verifier
        self.ver_threshold = ver_threshold
        # Ensemble (opt-in) settings. When surrogate_keys is None the original
        # single-model path is used and behaves exactly as before.
        self.surrogate_keys = surrogate_keys
        self.cross_model_aggregation = cross_model_aggregation
        self.model_weights = model_weights
        self.ensemble = surrogate_keys is not None
        self.ver_models: Optional[List] = None
        self.ref_embs: Optional[List[torch.Tensor]] = None

        if self.ensemble:
            validate_surrogate_keys(surrogate_keys)
            self.ver_models = []
            for key in surrogate_keys:
                name, variant = parse_model_key(key)
                self.ver_models.append(
                    get_verification_model(name, device=self.device, variant=variant)
                )
            # Keep self.ver_model pointing at the first surrogate for any code
            # that still references it; the single-model path is not used.
            self.ver_model = self.ver_models[0]
        else:
            self.ver_model = get_verification_model(embedder_name, device=self.device)

        # Prepare avatar
        self.targeted_features = targeted_features
        self.gaussians: FlameGaussianModel | None = None
        target_image_path = (
            target_image if isinstance(target_image, str) else target_image.as_posix()
        )
        if self.ensemble:
            ref_image = load_image_from_file(target_image_path)
            ref_image = ref_image.permute(1, 2, 0).to(self.device)  # (H, W, C)
            self.ref_embs = precompute_reference_embeddings(
                self.ver_models, ref_image
            )
            # Keep self.ref_emb for parity (first surrogate's reference).
            self.ref_emb = self.ref_embs[0]
        else:
            self.ref_emb = self._get_reference_id_embedding(image_path=target_image_path)

        # Prepare attack
        self.epsilons = epsilons
        self.adv_attack = adv_attack
        self.attack_steps = attack_steps
        # Optional step checkpointing.
        self.checkpoint_steps = (
            sorted({int(s) for s in checkpoint_steps}) if checkpoint_steps else None
        )
        # Optional extra eval viewpoints (orbit_x, orbit_y, orbit_z) rendered per
        # checkpoint so masking can be scored across views, not just frontally.
        self.eval_viewpoints = eval_viewpoints
        self.mask: torch.Tensor | None = None
        self.selected_regions = selected_regions
        self.att_tensor = self._prepare_tensor_for_attack(avatar_dir)
        self.foolbox_model, self.wrapped_module = self.setup_foolbox_attack()

        # Adaptive epsilon settings
        self.adaptive_epsilon = adaptive_epsilon
        self.region_multipliers = region_multipliers

        if self.checkpoint_steps and not self.adaptive_epsilon:
            raise ValueError(
                "checkpoint_steps is currently only supported with adaptive_epsilon=True."
            )

        # Output settings
        self.avatar_dir = avatar_dir
        # In ensemble mode the single `embedder_name` is not the surrogate that
        # masking optimizes against, so tag the run as an ensemble instead.
        tag = "ensemble" if self.ensemble else embedder_name
        self.output_base_name = f"{output_name}_{tag}_"

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

        if orbit_cam is not None:
            orbit_x, orbit_y, orbit_z = orbit_cam
            if orbit_x != 0:
                self.root_cam.orbit_x(orbit_x)
            if orbit_y != 0:
                self.root_cam.orbit_y(orbit_y)
            if orbit_z != 0:
                self.root_cam.orbit_z(orbit_z)

        original_features = get_targeted_features(self.gaussians, self.targeted_features)
        features = original_features.clone()
        new_features = new_features.to(features.device)
        features[self.mask] = new_features
        set_targeted_features(self.gaussians, self.targeted_features, features)
        try:
            rgb = render_single_frame(self.gaussians, self.root_cam, self.pipeline)
        finally:
            set_targeted_features(
                self.gaussians, self.targeted_features, original_features
            )

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
        if self.ensemble:
            verifier_model = FaceRenderVerification(
                embedders=self.ver_models,
                reference_embeddings=self.ref_embs,
                ver_threshold=self.ver_threshold,
                camera_boundary_angles=self.camera_angles,
                aggregation_mode=self.angle_aggregation,
                k=self.k_angles,
                render_fn=self.render_frame_in_rgb,
                cross_model_aggregation=self.cross_model_aggregation,
                model_weights=self.model_weights,
            )
        else:
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

    def save_results(
        self,
        adv_features: torch.Tensor,
        step: Optional[int] = None,
        save_ply: bool = True,
    ) -> None:
        if self.selected_regions:
            regions_str = "_".join(sorted(self.selected_regions))
        else:
            regions_str = "all"
        base_name = f"{self.output_base_name}{regions_str}"

        avatar_id = (
            Path(self.avatar_dir).name
            if isinstance(self.avatar_dir, str)
            else self.avatar_dir.name
        )

        output_name = DATASETS_DIR / f"seed{self.seed}" / base_name

        for eps, features in zip(self.epsilons, adv_features):
            # Per-eps directory, optionally nested under a per-step subfolder.
            eps_dir = output_name / f"eps_{eps:.3f}"
            if step is not None:
                eps_dir = eps_dir / f"step_{step:03d}"
            (eps_dir / "renders").mkdir(parents=True, exist_ok=True)

            with torch.no_grad():
                adv_rgb = self.render_frame_in_rgb(features)

            render_path = eps_dir / "renders" / f"{avatar_id}.png"
            adv_img_np = (np.clip(adv_rgb.cpu().detach().numpy(), 0, 1) * 255).astype(
                np.uint8
            )
            Image.fromarray(adv_img_np).save(render_path)

            # Multi-view renders: score masking across viewpoints
            if self.eval_viewpoints:
                mv_dir = eps_dir / "renders_mv" / avatar_id
                mv_dir.mkdir(parents=True, exist_ok=True)
                for vid, orbit in enumerate(self.eval_viewpoints):
                    with torch.no_grad():
                        view_rgb = self.render_frame_in_rgb(features, orbit_cam=orbit)
                    view_np = (
                        np.clip(view_rgb.cpu().detach().numpy(), 0, 1) * 255
                    ).astype(np.uint8)
                    ox, oy, oz = orbit
                    fname = f"v{vid:03d}_x{ox:+.3f}_y{oy:+.3f}_z{oz:+.3f}.png"
                    Image.fromarray(view_np).save(mv_dir / fname)

            if save_ply and self.targeted_features == "DC":
                if self.gaussians is None or self.mask is None:
                    raise RuntimeError(
                        "Gaussians and mask must be available to save PLY."
                    )
                orig_ply_path = Path(self.avatar_dir) / "point_cloud.ply"
                orig_flame_path = Path(self.avatar_dir) / "flame_param.npz"
                ply_output_path = eps_dir / "avatars" / avatar_id / "point_cloud.ply"
                ply_output_path.parent.mkdir(parents=True, exist_ok=True)
                new_features = get_targeted_features(
                    self.gaussians, self.targeted_features
                ).clone()
                new_features[self.mask] = features.to(new_features.device)
                write_ply_with_dc_colors(
                    original_ply_path=orig_ply_path,
                    new_colors=new_features,
                    output_ply_path=ply_output_path,
                )

                # Copy FLAME parameters if exist
                if orig_flame_path.exists():
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

        if self.adaptive_epsilon and self.checkpoint_steps:
            # per-step checkpointing: save one render set per step.
            step_to_adv = self._run_adaptive_epsilon_attack(target_class)
            max_step = max(self.checkpoint_steps)
            for step in self.checkpoint_steps:
                # Only persist the PLY/FLAME files at the final step.
                self.save_results(
                    step_to_adv[step], step=step, save_ply=(step == max_step)
                )
            clipped_adv = step_to_adv[max_step]
        else:
            if self.adaptive_epsilon:
                # Use custom PGD with per-element epsilon
                clipped_adv = self._run_adaptive_epsilon_attack(target_class)
            else:
                # Use standard Foolbox attack
                clipped_adv = self._run_foolbox_attack(target_class)

            self.save_results(clipped_adv)

        for eps, adv_features in zip(self.epsilons, clipped_adv):
            with torch.no_grad():
                adv_sim = self.wrapped_module.compute_similarity(
                    adv_features.to(self.device)
                ).item()
            print(f"Epsilon: {eps:.3f} -> Aggregated Cosine Similarity: {adv_sim:.4f}")

    def _run_foolbox_attack(self, target_class: torch.Tensor) -> List[torch.Tensor]:
        """Run standard Foolbox attack with uniform epsilon."""
        attack = get_foolbox_attack(
            adv_attack_name=self.adv_attack,
            steps=self.attack_steps,
            random_start=False,
        )

        # Create progress bar for attack steps
        num_epsilons = 1 if self.adv_attack.lower() == "ddn" else len(self.epsilons)
        total_steps = self.attack_steps * num_epsilons

        # Wrap the model to track forward passes
        original_forward = self.wrapped_module.forward
        pbar = tqdm(total=total_steps, desc="Foolbox Attack Steps", unit="step")

        def forward_with_progress(*args, **kwargs):
            result = original_forward(*args, **kwargs)
            pbar.update(1)
            return result

        self.wrapped_module.forward = forward_with_progress

        try:
            epsilons = None if self.adv_attack.lower() == "ddn" else self.epsilons
            raw_adv, clipped_adv, is_adv = attack(
                model=self.foolbox_model,
                inputs=self.att_tensor.unsqueeze(0),
                criterion=target_class,
                epsilons=epsilons,
            )
        finally:
            # Restore original forward method
            self.wrapped_module.forward = original_forward
            pbar.close()

        return clipped_adv

    def _run_adaptive_epsilon_attack(self, target_class: torch.Tensor):
        """Run custom PGD attack with per-Gaussian adaptive epsilon.

        Returns a ``list`` of adversarial tensors (one per epsilon) by default.
        When ``self.checkpoint_steps`` is set, returns a ``{step: [per-eps
        tensors]}`` dict so each checkpoint can be saved separately.
        """
        if self.gaussians is None:
            raise RuntimeError("Gaussians must be loaded before running attack.")

        clipped_adv: List[torch.Tensor] = []
        step_to_adv: dict[int, List[torch.Tensor]] = (
            {step: [] for step in self.checkpoint_steps}
            if self.checkpoint_steps
            else {}
        )
        for base_eps in tqdm(self.epsilons, desc="Processing epsilons", unit="eps"):
            # Create per-Gaussian epsilon mask
            full_eps = create_adaptive_epsilon_mask(
                gaussians=self.gaussians,
                base_epsilon=base_eps,
                region_multipliers=self.region_multipliers,
            )

            # Apply mask to get epsilon for attacked Gaussians only
            eps_for_attack = full_eps[self.mask]

            # Run adaptive PGD attack
            adv_features = adaptive_linf_pgd_attack(
                model=self.wrapped_module,
                inputs=self.att_tensor.to(self.device),
                epsilon_per_element=eps_for_attack,
                target_class=target_class,
                steps=self.attack_steps,
                random_start=False,
                checkpoint_steps=self.checkpoint_steps,
            )
            if self.checkpoint_steps:
                for step in self.checkpoint_steps:
                    step_to_adv[step].append(adv_features[step])
            else:
                clipped_adv.append(adv_features)

        return step_to_adv if self.checkpoint_steps else clipped_adv
