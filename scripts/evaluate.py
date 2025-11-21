import os
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
warnings.filterwarnings("ignore", message="Protobuf gencode version")

from aegis.evaluation.datasets import (
    DatasetIdentityLookup,
    FaceDataset,
    GallerySource,
    CompositeIdentityLookup,
    resolve_dataset,
)
from aegis.evaluation.evaluators import (
    RankKEvaluator,
    UtilityEvaluator,
    compute_or_load_verification_threshold,
)
from aegis.evaluation.stores import (
    load_embeddings,
)
from aegis.models import AdaFaceEmbedder, ArcFaceEmbedder, resolve_compute_device
from aegis.utils import load_image_map, ensure_csv_parent
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

try:  # pragma: no cover - runtime fallback for script execution
    from ..src.aegis.config import ROOT_DIR
except ImportError:  # pragma: no cover
    PACKAGE_ROOT = Path(__file__).resolve().parent.parent
    if str(PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(PACKAGE_ROOT))
    from aegis.config import ROOT_DIR


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate anonymised face datasets")
    parser.add_argument(
        "--dataset",
        choices=["CelebA", "lfw", "NeRSembleGT"],
        required=False,
        help="(deprecated) Primary dataset used in older invocations; prefer --gallery-dataset",
    )
    parser.add_argument(
        "--anon-path",
        type=Path,
        default=None,
        help="(deprecated) Path to the anonymised dataset root; use --query-path instead",
    )
    parser.add_argument(
        "--query-path",
        type=Path,
        default=None,
        help="Path to the query (typically anonymised) dataset root",
    )
    parser.add_argument(
        "--gallery-dataset",
        dest="gallery_datasets",
        action="append",
        choices=["CelebA", "lfw", "NeRSembleGT"],
        default=None,
        help="Datasets to enrol as the gallery (repeatable)",
    )
    parser.add_argument(
        "--anonymized-dataset",
        choices=["CelebA", "lfw", "NeRSembleGT", "NeRSembleReconst"],
        default=None,
        help="Dataset definition whose anonymised renders are supplied",
    )
    parser.add_argument(
        "--anonymized-path",
        type=Path,
        default=None,
        help="Directory containing the anonymised renders",
    )
    parser.add_argument(
        "--anonymized-label",
        type=str,
        default=None,
        help="Label used when caching/reporting anonymised embeddings (defaults to dataset name + '_anon')",
    )
    parser.add_argument(
        "--anonymized-extension",
        type=str,
        default=None,
        help="Override the expected file extension for anonymised renders",
    )
    parser.add_argument(
        "--label",
        type=str,
        default=None,
        help="Label used when writing evaluation results (defaults to folder name)",
    )
    parser.add_argument(
        "--celeba-test-set-only",
        action="store_true",
        help="When evaluating CelebA, restrict to the official test split",
    )
    parser.add_argument(
        "--gallery-extra",
        action="append",
        choices=["CelebA", "lfw", "NeRSembleGT"],
        default=[],
        help="(deprecated) Additional datasets to enrol into the gallery (use --gallery-dataset instead)",
    )
    parser.add_argument(
        "--query-source",
        choices=["CelebA", "lfw", "NeRSembleGT"],
        default=None,
        help="(deprecated) Dataset identity for the query set; use --anonymized-dataset instead",
    )
    parser.add_argument(
        "--evaluation-method",
        choices=["rank_k", "verification", "utility"],
        default="rank_k",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--embedder", choices=["arcface", "adaface"], default="arcface")
    parser.add_argument(
        "--query-extension",
        type=str,
        default=None,
        help="Override the expected file extension for query images (include leading dot)",
    )
    parser.add_argument(
        "--adaface-model-path",
        type=Path,
        default=ROOT_DIR / "models",
        help="Directory containing AdaFace checkpoints",
    )
    parser.add_argument(
        "--adaface-model-type",
        choices=["ir50", "ir101"],
        default="ir50",
        help="Which AdaFace backbone to use",
    )
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--rank-k-one-image-per", action="store_true")
    parser.add_argument("--num-verification-pairs", type=int, default=5000)
    parser.add_argument(
        "--verification-threshold-dataset",
        choices=["CelebA", "lfw", "NeRSembleGT"],
        default=None,
        help="Dataset used to fit the verification decision threshold (defaults to --anonymized-dataset).",
    )
    parser.add_argument(
        "--force-threshold-recompute",
        action="store_true",
        help="Force recomputation of verification threshold even if a cached result exists.",
    )
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "output" / "evaluations",
        help="Root directory where evaluation artefacts are written",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    anonymized_path = args.anonymized_path or args.query_path or args.anon_path
    if anonymized_path is None:
        parser.error(
            "You must supply --anonymized-path (or legacy --query-path/--anon-path)."
        )
    if (
        args.anon_path is not None
        and args.query_path is None
        and args.anonymized_path is None
    ):
        print(
            "[evaluate] --anon-path is deprecated, please use --anonymized-path instead.",
            file=sys.stderr,
        )

    gallery_dataset_names: List[str] = []
    if args.gallery_datasets:
        gallery_dataset_names.extend(args.gallery_datasets)
    if args.dataset:
        gallery_dataset_names.append(args.dataset)
    gallery_dataset_names.extend(args.gallery_extra or [])

    if not gallery_dataset_names and not args.evaluation_method == "utility":
        parser.error(
            "Please provide at least one gallery dataset via --gallery-dataset or --dataset."
        )

    # Deduplicate while preserving order
    seen_gallery: set[str] = set()
    ordered_gallery: List[str] = []
    for name in gallery_dataset_names:
        if name not in seen_gallery:
            ordered_gallery.append(name)
            seen_gallery.add(name)
    gallery_dataset_names = ordered_gallery

    anonymized_dataset_name = (
        args.anonymized_dataset
        or args.query_source
        or (gallery_dataset_names[0] if len(gallery_dataset_names) == 1 else None)
    )
    if anonymized_dataset_name is None:
        parser.error(
            "Specify --anonymized-dataset when multiple gallery datasets are provided."
        )
    if (
        anonymized_dataset_name not in gallery_dataset_names
        and not args.evaluation_method == "utility"
    ):
        parser.error(
            "Anonymized dataset must also be included in the gallery datasets list."
        )

    gallery_sources: List[GallerySource] = []
    for name in gallery_dataset_names:
        spec = resolve_dataset(
            name, args.celeba_test_set_only if name == "CelebA" else False
        )
        dataset = FaceDataset(
            spec.images_root,
            spec.file_extension,
            celeba_test_set_only=spec.celeba_test_set_only,
        )
        cache_dir = args.cache_dir or spec.cache_dir
        cache_dir.mkdir(parents=True, exist_ok=True)
        gallery_sources.append(
            GallerySource(
                name=spec.name,
                prefix=spec.name,
                dataset=dataset,
                dataset_root=spec.root,
                images_root=spec.images_root,
                identity_lookup=spec.identity_lookup,
                cache_dir=cache_dir,
            )
        )

    anonymized_spec = resolve_dataset(
        anonymized_dataset_name,
        args.celeba_test_set_only if anonymized_dataset_name == "CelebA" else False,
    )
    anonymized_extension = (
        args.anonymized_extension
        or args.query_extension
        or anonymized_spec.file_extension
    )
    anonymized_dataset = FaceDataset(anonymized_path, anonymized_extension)
    anonymized_cache_dir = args.cache_dir or (Path(anonymized_path).parent / ".cache")
    anonymized_cache_dir.mkdir(parents=True, exist_ok=True)
    anonymized_label = (
        args.anonymized_label
        or Path(anonymized_path).name
        or f"{anonymized_spec.name}_anon"
    )

    label = args.label or anonymized_label

    device = resolve_compute_device(args.device)
    if args.embedder == "arcface":
        cache_suffix = "arcface"
        embedder = ArcFaceEmbedder(device=device, batch_size=args.batch_size)
    elif args.embedder == "adaface":
        cache_suffix = f"adaface_{args.adaface_model_type}"
        embedder = AdaFaceEmbedder(
            device=device,
            batch_size=args.batch_size,
            model_path=args.adaface_model_path,
            model_type=args.adaface_model_type,
        )
    else:  # pragma: no cover - defensive programming
        raise ValueError(f"Unsupported embedder {args.embedder}")

    identity_mapping: Dict[str, DatasetIdentityLookup] = {}
    gallery_embeddings: Dict[str, np.ndarray] = {}

    for source in gallery_sources:
        identity_mapping[source.prefix] = source.identity_lookup
        cache_key = source.prefix
        cache_path = source.cache_dir / f"{cache_key}_{cache_suffix}.pkl"
        embeddings = load_embeddings(
            embedder,
            source.dataset,
            source.images_root,
            cache_path,
            key_prefix=source.prefix,
        )
        gallery_embeddings.update(embeddings)

    identity_mapping.setdefault(anonymized_spec.name, anonymized_spec.identity_lookup)
    anonymized_cache_key = anonymized_label
    query_cache_path = (
        anonymized_cache_dir / f"{anonymized_cache_key}_{cache_suffix}.pkl"
    )
    query_embeddings = load_embeddings(
        embedder,
        anonymized_dataset,
        anonymized_path,
        query_cache_path,
        key_prefix=anonymized_spec.name,
        load_from_cache=False,  # temp solution for issues with cache loading
    )

    if args.evaluation_method == "utility":
        # For utility evaluation, we do not need to build a composite lookup
        composite_lookup = anonymized_spec.identity_lookup
    else:
        composite_lookup = CompositeIdentityLookup(
            identity_mapping, default_lookup=gallery_sources[0].identity_lookup
        )

        if not gallery_embeddings:
            raise RuntimeError(
                "Gallery enrollment produced no embeddings; check gallery sources."
            )
        if not query_embeddings:
            raise RuntimeError(
                "Anonymised set produced no embeddings; verify the path and extension."
            )

    result_dataset_name = anonymized_spec.name

    if args.evaluation_method == "rank_k":
        evaluator = RankKEvaluator(
            identity_lookup=composite_lookup,
            real_embeddings=gallery_embeddings,
            anon_embeddings=query_embeddings,
            limit_one_per_identity=args.rank_k_one_image_per,
        )
        df = evaluator.run()
        if args.label:
            out_path = args.output_dir / args.label / "rank_k.csv"
        else:
            out_path = args.output_dir / result_dataset_name / label / "rank_k.csv"
        ensure_csv_parent(out_path)
        df.to_csv(out_path, index=False)
        report_rank_k(df)
        return

    if args.evaluation_method == "utility":
        real_source = anonymized_spec.images_root
        real_images, real_paths = load_image_map(
            os.listdir(real_source),
            real_source,
            key_prefix=anonymized_spec.name,
        )
        if not real_images:
            raise RuntimeError("Failed to load any real images for utility evaluation.")

        anon_images, anon_paths = load_image_map(
            [path.name for path in anonymized_dataset.paths],
            str(anonymized_dataset.root),
            key_prefix=anonymized_spec.name,
        )
        if not anon_images:
            raise RuntimeError(
                "Failed to load any anonymized images for utility evaluation."
            )

        evaluator = UtilityEvaluator(
            identity_lookup=composite_lookup,
            real_images=real_images,
            real_paths=real_paths,
            anon_images=anon_images,
            anon_paths=anon_paths,
        )
        df = evaluator.run()
        if args.label:
            out_path = args.output_dir / args.label / "utility.csv"
        else:
            out_path = args.output_dir / result_dataset_name / label / "utility.csv"
        ensure_csv_parent(out_path)
        df.to_csv(out_path, index=False)
        report_utility(df)
        return

    if args.evaluation_method == "verification":
        threshold_dataset_name = (
            args.verification_threshold_dataset or anonymized_spec.name
        )
        threshold_source = next(
            (src for src in gallery_sources if src.name == threshold_dataset_name), None
        )

        if threshold_source:
            prefix_token = f"{threshold_source.prefix}___"
            threshold_embeddings = {
                key: emb
                for key, emb in gallery_embeddings.items()
                if key.startswith(prefix_token)
            }
            threshold_identity_lookup = threshold_source.identity_lookup
            threshold_cache_dir = threshold_source.cache_dir
            if not threshold_embeddings:
                cache_path = (
                    threshold_source.cache_dir
                    / f"{threshold_source.prefix}_{cache_suffix}.pkl"
                )
                threshold_embeddings = load_embeddings(
                    embedder,
                    threshold_source.dataset,
                    threshold_source.images_root,
                    cache_path,
                    key_prefix=threshold_source.prefix,
                )
                gallery_embeddings.update(threshold_embeddings)
        else:
            threshold_spec = resolve_dataset(
                threshold_dataset_name,
                (
                    args.celeba_test_set_only
                    if threshold_dataset_name == "CelebA"
                    else False
                ),
            )
            threshold_dataset = FaceDataset(
                threshold_spec.images_root,
                threshold_spec.file_extension,
                celeba_test_set_only=threshold_spec.celeba_test_set_only,
            )
            threshold_cache_dir = args.cache_dir or threshold_spec.cache_dir
            threshold_cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = (
                threshold_cache_dir / f"{threshold_spec.name}_{cache_suffix}.pkl"
            )
            threshold_embeddings = load_embeddings(
                embedder,
                threshold_dataset,
                threshold_spec.images_root,
                cache_path,
                key_prefix=threshold_spec.name,
            )
            threshold_identity_lookup = threshold_spec.identity_lookup

        if not threshold_embeddings:
            raise RuntimeError(
                f"No embeddings available for threshold dataset '{threshold_dataset_name}'. Make sure it is enrolled or cached."
            )

        thresholds_root = threshold_cache_dir / "verification_thresholds"
        threshold_result = compute_or_load_verification_threshold(
            dataset_name=threshold_dataset_name,
            embedder_key=cache_suffix,
            embeddings=threshold_embeddings,
            identity_lookup=threshold_identity_lookup,
            thresholds_root=thresholds_root,
            num_pairs=args.num_verification_pairs,
            random_seed=args.random_seed,
            force_recompute=args.force_threshold_recompute,
        )

        print(
            f"Using verification threshold {threshold_result.eer_score_threshold:.4f} derived from {threshold_dataset_name} "
            f"(EER {threshold_result.eer:.2%}; cache: {threshold_result.cache_file})."
        )
        if threshold_result.plot_path.exists():
            print(f"FAR/FRR curve saved to {threshold_result.plot_path}.")

        # 1. Get the prefix for the dataset that was anonymized
        anon_dataset_prefix = f"{anonymized_spec.name}___"

        # 2. Filter the gallery to get *only* the real embeddings for that dataset
        real_counterpart_embeddings = {
            key: emb
            for key, emb in gallery_embeddings.items()
            if key.startswith(anon_dataset_prefix)
        }

        if not real_counterpart_embeddings:
            raise RuntimeError(
                f"Could not find any 'real' embeddings in the gallery "
                f"matching the prefix '{anon_dataset_prefix}'. "
                f"Ensure '{anonymized_spec.name}' is in the gallery datasets."
            )

        # 3. Get the *specific* identity lookup for that dataset
        real_counterpart_lookup = identity_mapping.get(anonymized_spec.name)
        if real_counterpart_lookup is None:
            raise RuntimeError(
                f"Internal error: No identity lookup found for '{anonymized_spec.name}'"
            )

        print(f"Evaluating verification for {anonymized_spec.name}:")
        print(
            f"  Gallery (real counterparts): {len(real_counterpart_embeddings)} embeddings"
        )
        print(f"  Query (anonymized): {len(query_embeddings)} embeddings")

        # 4. Get the lookup for the anonymized dataset
        # We assume the anonymized dataset uses the same identity logic as its real counterpart
        anon_identity_lookup = anonymized_spec.identity_lookup
        if anon_identity_lookup is None:
            raise RuntimeError(
                f"Internal error: No identity lookup for anonymized spec '{anonymized_spec.name}'"
            )

        # 5. Map identity -> list of embeddings for both real and anon
        real_id_to_emb = {}
        for key, emb in real_counterpart_embeddings.items():
            try:
                identity = real_counterpart_lookup.lookup(key)
                if identity not in real_id_to_emb:
                    real_id_to_emb[identity] = []
                real_id_to_emb[identity].append(emb)
            except Exception:
                continue

        anon_id_to_emb = {}
        for key, emb in query_embeddings.items():
            try:
                identity = anon_identity_lookup.lookup(key)
                if identity not in anon_id_to_emb:
                    anon_id_to_emb[identity] = []
                anon_id_to_emb[identity].append(emb)
            except Exception:
                continue

        print(f"  Found {len(real_id_to_emb)} real identities.")
        print(f"  Found {len(anon_id_to_emb)} anonymized identities.")

        # 6. Build pairs by comparing all real against all anonymized
        rows = []
        threshold = threshold_result.eer_score_threshold

        all_anon_identities = list(anon_id_to_emb.keys())

        # Iterate over all anon embeddings for each identity
        for anon_id in all_anon_identities:
            anon_embeds = anon_id_to_emb[anon_id]
            if anon_id not in real_id_to_emb:
                print(f"Warning: id '{anon_id}' missing from real_id_to_emb")
                continue  # Should not happen, but good to check

            real_embeds = real_id_to_emb[anon_id]

            # check each of anon embeddings against all real embeddings of the same id
            for anon_emb_idx, anon_emb in enumerate(anon_embeds):
                anon_emb_tensor = torch.tensor(anon_emb)
                real_embeds_tensor = torch.tensor(real_embeds)

                similarity = torch.clip(
                    torch.cosine_similarity(real_embeds_tensor, anon_emb_tensor, dim=1),
                    min=-1.0,
                    max=1.0,
                )
                # take max similarity and its idx
                max_sim_idx = torch.argmax(similarity).item()
                max_similarity = similarity[max_sim_idx].item()
                pred = 1 if max_similarity >= threshold else 0

                rows.append(
                    {
                        "subject_id": anon_id,
                        "emb_idx": anon_emb_idx,
                        "match": True if pred == 1 else False,
                        "max_similarity": max_similarity,
                    }
                )

        if not rows:
            raise RuntimeError(
                "Failed to perform verification. Check identity lookups."
            )

        df = pd.DataFrame(rows)

        # 7. Report results
        if args.label:
            out_path = args.output_dir / args.label / "verification.csv"
        else:
            out_path = (
                args.output_dir / result_dataset_name / label / "verification.csv"
            )
        ensure_csv_parent(out_path)
        df.to_csv(out_path, index=False)
        report_verification(
            df,
            threshold_result.distance_threshold,
            threshold_result.eer,
            threshold_result.eer_score_threshold,
        )
        return

    raise ValueError(f"Unknown evaluation method {args.evaluation_method}")


def report_rank_k(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results to report for rank-k evaluation")
        return
    print("================ Rank-K Results ================")
    for k in [1, 5, 10, 20, 50]:
        coverage = (df["k"] < k).mean()
        print(f"Accuracy @ k={k:02d}: {coverage:.2%}")
    print(f"Detection rate: {len(df)} samples")


def report_utility(df: pd.DataFrame) -> None:
    if df.empty:
        print("No results to report for utility evaluation")
        return
    print("================ Utility Results ================")
    for metric in [
        "ssim",
        "psnr",
        "age_diff",
        "emotion_match",
        "gender_match",
        "race_match",
    ]:
        if metric in df.columns:
            value = df[metric].mean()
            if metric == "age_diff":
                print(f"Average Age Difference: {value:.2f} years")
            elif metric in ["ssim", "psnr"]:
                print(f"Average {metric.upper()}: {value:.4f}")
            else:
                print(f"Average {metric.replace('_', ' ').title()}: {value:.2%}")


def report_verification(
    df: pd.DataFrame,
    threshold: float,
    fitted_eer: Optional[float] = None,
    fitted_eer_score: Optional[float] = None,
) -> None:
    if df.empty:
        print("No verification pairs available")
        return
    accuracy = df["match"].mean()
    print("================ Verification Results ================")
    print(f"Threshold (fitted on real pairs): {threshold:.4f}")
    if fitted_eer is not None:
        print(f"Gallery EER (real pairs): {fitted_eer:.2%}")
    if fitted_eer_score is not None:
        print(f"EER score threshold (cosine similarity): {fitted_eer_score:.4f}")
    print(f"Match accuracy (on anonymized dataset): {accuracy:.2%}")


if __name__ == "__main__":
    main()
