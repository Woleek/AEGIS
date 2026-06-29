import argparse
import json
from aegis import AEGIS
from aegis.models import list_verification_models

_EMBEDDER_CHOICES = sorted(list_verification_models())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="AEGIS Avatar Identity Masking")
    parser.add_argument(
        "--avatar-dir",
        type=str,
        required=True,
        help="Path to the avatar directory containing point_cloud.ply.",
    )
    parser.add_argument(
        "--target-image",
        type=str,
        required=True,
        help="Path to the target image for verification reference.",
    )
    parser.add_argument(
        "--epsilons",
        type=float,
        nargs="+",
        default=[0.05, 0.1, 0.2, 0.3],
        help="List of epsilon values for the attack. Default is [0.05, 0.1, 0.2, 0.3].",
    )
    parser.add_argument(
        "--attack-steps",
        type=int,
        default=300,
        help="Number of attack steps to perform. Default is 300.",
    )
    parser.add_argument(
        "--ver-threshold",
        type=float,
        default=0.1720,
        help="Cosine similarity threshold for face verification. Defaults to 0.1720 for AdaFace (from Labeled Faces in the Wild dataset).",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="adaface",
        choices=_EMBEDDER_CHOICES,
        help=f"Face verification model to use. Choices: {', '.join(_EMBEDDER_CHOICES)}. Default is 'adaface'.",
    )
    parser.add_argument(
        "--select-regions",
        type=str,
        nargs="+",
        default=[],
        help="Regions to select for attack (e.g., eyes, lips, nose, ears, forehead). Default is all regions.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default is 42.",
    )
    parser.add_argument(
        "--camera-boundary-angles",
        type=float,
        nargs="+",
        default=None,
        help=(
            "Boundary camera angles used for sampling during attack. "
            "Provide as a list of floats in the order: orbit_x_min orbit_x_max "
            "orbit_y_min orbit_y_max orbit_z_min orbit_z_max. "
            "If not provided, defaults to no transformation."
        ),
    )
    parser.add_argument(
        "--angle-aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "min", "median"],
        help=(
            "How to aggregate per-view cosine similarities when multiple camera angles "
            "are used. Default is 'mean'."
        ),
    )
    parser.add_argument(
        "--k-angles",
        type=int,
        default=5,
        help="Number of camera angles to sample per iteration within the specified boundaries. Default is 5.",
    )
    parser.add_argument(
        "--target-features",
        type=str,
        default="DC",
        help="Type of features to target for the attack. Default is 'DC' (base color).",
        choices=["DC", "AC", "pos", "scale", "rot", "opacity"],
    )
    parser.add_argument(
        "--adv-attack",
        type=str,
        default="linfpgd",
        help="Adversarial attack method to use. Default is 'linfpgd'.",
        choices=[
            "linfpgd",
            "l2pgd",
            "linffgsm",
            "l2fgsm",
            "ddn",
        ],
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="NeRSembleMasked",
        help="Dataset name for output files. Default is 'NeRSembleMasked'.",
    )
    parser.add_argument(
        "--adaptive-epsilon",
        action="store_true",
        help=("Enable adaptive regional epsilon budgets."),
    )
    parser.add_argument(
        "--region-multipliers",
        type=str,
        default=None,
        help=(
            "JSON string of region->multiplier mappings for adaptive epsilon. "
            'Example: \'{"skin": 0.5, "eyes": 1.2, "nose": 1.0}\'. '
            "If not provided, uses default multipliers."
        ),
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1,
        help="Camera orbit radius. Use 1 for NeRSemble avatars, 20 for FaceScape. Default is 1.",
    )
    parser.add_argument(
        "--surrogate-keys",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Opt-in FR ensemble: space-separated surrogate model keys ('model' or "
            "'model:variant', e.g. arcface:r50 facenet:vggface2 swinface:swin_t). "
            "When omitted, single-model masking with --embedder is used."
        ),
    )
    parser.add_argument(
        "--cross-model-aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "min", "median"],
        help="How to aggregate per-model similarities in ensemble mode. Default is 'mean'.",
    )
    parser.add_argument(
        "--model-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional per-surrogate weights (same count/order as --surrogate-keys).",
    )
    parser.add_argument(
        "--checkpoint-steps",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Step counts at which to also save intermediate masked results "
            "(e.g. 50 100 150 200 250 300). Only valid with --adaptive-epsilon currently."
        ),
    )
    parser.add_argument(
        "--eval-view-grid",
        type=int,
        default=None,
        help=(
            "If set, render a K x K grid of extra eval viewpoints per saved "
            "checkpoint (spanning --camera-boundary-angles in orbit_x/orbit_y, "
            "z fixed at its midpoint) into renders_mv/. Frontal render is always saved separately."
        ),
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Parse region multipliers JSON if provided
    region_multipliers = None
    if args.region_multipliers:
        region_multipliers = json.loads(args.region_multipliers)

    # Build the K x K eval viewpoint grid (orbit_x, orbit_y) over the camera
    # boundary angles, z fixed at its midpoint.
    eval_viewpoints = None
    if args.eval_view_grid and args.eval_view_grid > 0:
        if not args.camera_boundary_angles or len(args.camera_boundary_angles) < 6:
            parser_error = (
                "--eval-view-grid requires --camera-boundary-angles with 6 values "
                "(x_min x_max y_min y_max z_min z_max)."
            )
            raise SystemExit(parser_error)
        import numpy as np

        x_min, x_max, y_min, y_max, z_min, z_max = args.camera_boundary_angles[:6]
        k = args.eval_view_grid
        xs = np.linspace(x_min, x_max, k)
        ys = np.linspace(y_min, y_max, k)
        z_mid = (z_min + z_max) / 2.0
        eval_viewpoints = [
            (float(ox), float(oy), float(z_mid)) for ox in xs for oy in ys
        ]

    AEGIS(
        embedder_name=args.embedder,
        avatar_dir=args.avatar_dir,
        target_image=args.target_image,
        epsilons=args.epsilons,
        selected_regions=args.select_regions,
        camera_boundary_angles=args.camera_boundary_angles,
        angle_aggregation=args.angle_aggregation,
        k_angles=args.k_angles,
        targeted_features=args.target_features,
        adv_attack=args.adv_attack,
        attack_steps=args.attack_steps,
        ver_threshold=args.ver_threshold,
        seed=args.seed,
        output_name=args.output_name,
        adaptive_epsilon=args.adaptive_epsilon,
        region_multipliers=region_multipliers,
        radius=args.radius,
        surrogate_keys=args.surrogate_keys,
        cross_model_aggregation=args.cross_model_aggregation,
        model_weights=args.model_weights,
        checkpoint_steps=args.checkpoint_steps,
        eval_viewpoints=eval_viewpoints,
    ).run()
