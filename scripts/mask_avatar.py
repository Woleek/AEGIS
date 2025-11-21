import argparse
from aegis import AEGIS


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
        choices=["arcface", "adaface"],
        help="Face verification model to use. Choices are 'arcface' or 'adaface'. Default is 'adaface'.",
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
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
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
    ).run()
