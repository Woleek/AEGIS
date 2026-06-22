#!/usr/bin/env python
"""Single-view single-reference avatar verification (1:1).

Renders the avatar from a frontal view (or loads a pre-rendered frontal image), 
embeds it with the chosen face-recognition model, and compares its cosine similarity 
against the first image of a ground-truth gallery directory using the model's calibrated
threshold (or an explicit ``--threshold``).

Prints VERIFIES when ``cosine > threshold`` and FAILS otherwise. If no face is
detected in the rendered view, the result is reported as N/A.
"""

import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", message="Protobuf gencode version")

import argparse
from pathlib import Path
from typing import Optional

import torch

try:  # pragma: no cover - runtime fallback for script execution
    from ..src.aegis.config import ROOT_DIR
except ImportError:  # pragma: no cover
    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))

from aegis.models import (
    FaceNotDetectedError,
    get_model_spec,
    get_verification_model,
    list_verification_models,
    resolve_compute_device,
)
from aegis.splat import PipelineConfig, load_gaussians, render_single_frame
from aegis.utils import load_image_from_file, seed_experiment

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Single-view single-reference avatar verification (1:1)."
    )
    parser.add_argument(
        "--gt-images",
        type=Path,
        required=True,
        help="Directory of ground-truth identity images. Only the first image is used as reference.",
    )
    parser.add_argument(
        "--avatar-dir",
        type=Path,
        default=None,
        help="Avatar directory containing point_cloud.ply. Required unless --render-image is given.",
    )
    parser.add_argument(
        "--render-image",
        type=Path,
        default=None,
        help="Pre-rendered frontal image (PNG/JPG). If set, skips avatar loading/rendering.",
    )
    parser.add_argument(
        "--embedder",
        choices=sorted(list_verification_models()),
        default="arcface",
        help="Face verification model.",
    )
    parser.add_argument(
        "--embedder-variant",
        type=str,
        default=None,
        help="Optional model variant (e.g. ir50, r50). Uses the model default when omitted.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Cosine similarity threshold. Defaults to the model spec's calibrated threshold.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.0,
        help="Camera orbit radius. Use 1 for NeRSemble avatars, 20 for FaceScape.",
    )
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def first_gallery_image(gt_dir: Path) -> Path:
    if not gt_dir.exists() or not gt_dir.is_dir():
        raise FileNotFoundError(f"GT image directory not found: {gt_dir}")
    paths = sorted(
        p for p in gt_dir.iterdir() if p.suffix.lower() in _IMAGE_EXTENSIONS
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {gt_dir}")
    return paths[0]


def render_frontal(avatar_dir: Path, radius: float) -> torch.Tensor:
    """Render the avatar's frontal view as an (H, W, C) RGB tensor in [0, 1]."""
    from utils.viewer_utils import OrbitCamera

    gaussians = load_gaussians(point_path=avatar_dir / "point_cloud.ply")
    pipeline = PipelineConfig(background_color=[1.0, 1.0, 1.0])
    root_cam = OrbitCamera(960, 540, r=radius, fovy=20, convention="opencv")
    rgb = render_single_frame(gaussians, root_cam, pipeline)  # (H, W, C)
    return rgb.detach().clamp(0, 1)


def main(argv: Optional[list] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if (args.avatar_dir is None) == (args.render_image is None):
        parser.error("Specify exactly one of --avatar-dir or --render-image.")

    seed_experiment(args.seed)
    device = resolve_compute_device(args.device)

    model_spec = get_model_spec(args.embedder, args.embedder_variant)
    threshold = args.threshold if args.threshold is not None else model_spec.threshold
    if threshold is None:
        parser.error(
            f"No verification threshold configured for {model_spec.model_id}. "
            "Pass --threshold explicitly."
        )

    embedder = get_verification_model(
        args.embedder, device=device, variant=args.embedder_variant
    )
    print(f"Embedder loaded: {model_spec.model_id}")

    # Reference embedding (first image in the GT gallery).
    gt_path = first_gallery_image(args.gt_images)
    print(f"Gallery reference: {gt_path}")
    gt_img = load_image_from_file(str(gt_path)).permute(1, 2, 0).to(device)  # (H, W, C)
    with torch.no_grad():
        gt_emb = embedder(gt_img)

    # Avatar / rendered frontal view.
    if args.render_image is not None:
        if not args.render_image.exists():
            raise FileNotFoundError(f"Render image not found: {args.render_image}")
        avatar_label = str(args.render_image)
        img_hwc = (
            load_image_from_file(str(args.render_image)).permute(1, 2, 0).to(device)
        )
        print(f"Loaded pre-rendered frontal: {args.render_image.name}")
    else:
        avatar_label = str(args.avatar_dir)
        print(f"Rendering frontal view from {args.avatar_dir} ...")
        img_hwc = render_frontal(args.avatar_dir, args.radius).to(device)

    # Embed rendered view + cosine similarity vs reference.
    try:
        with torch.no_grad():
            adv_emb = embedder(img_hwc)
        cos_sim = float(torch.cosine_similarity(adv_emb, gt_emb, dim=1).item())
        face_detected = True
    except FaceNotDetectedError:
        cos_sim = float("nan")
        face_detected = False

    sim_str = "N/A" if not face_detected else f"{cos_sim:.4f}"
    print("======== Avatar Verification (single-view, 1:1) ========")
    print(f"  Avatar     : {avatar_label}")
    print(f"  Reference  : {gt_path}")
    print(f"  Embedder   : {model_spec.model_id}")
    print(f"  Threshold  : {threshold:.4f}")
    print(f"  Cosine sim : {sim_str}")

    if not face_detected:
        print("  Result     : N/A - no face detected in rendered frontal view")
    elif cos_sim > threshold:
        print(f"  Result     : VERIFIES  ({cos_sim:.4f} > {threshold:.4f})")
    else:
        print(f"  Result     : FAILS  ({cos_sim:.4f} <= {threshold:.4f})")


if __name__ == "__main__":
    main()
