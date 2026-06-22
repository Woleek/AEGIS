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
from aegis.models import (
    get_eval_embedder,
    resolve_compute_device,
)
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
        choices=["CelebA", "lfw", "NeRSembleGT", "FaceScapeGT", "CombinedGT"],
        default=None,
        help="Datasets to enrol as the gallery (repeatable)",
    )
    parser.add_argument(
        "--anonymized-dataset",
        choices=[
            "CelebA",
            "lfw",
            "NeRSembleGT",
            "NeRSembleReconst",
            "FaceScapeGT",
            "FaceScapeReconst",
            "CombinedGT",
            "CombinedReconst",
        ],
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
        choices=["CelebA", "lfw", "NeRSembleGT", "FaceScapeGT", "CombinedGT"],
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
    parser.add_argument(
        "--embedder",
        choices=["arcface", "adaface", "swinface", "transface", "facenet", "cosface", "ir152", "irse50", "mobileface"],
        default="arcface",
    )
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
        choices=["CelebA", "lfw", "NeRSembleGT", "FaceScapeGT"],
        default=None,
        help="Dataset used to fit the verification decision threshold (defaults to --anonymized-dataset).",
    )
    parser.add_argument(
        "--force-threshold-recompute",
        action="store_true",
        help="Force recomputation of verification threshold even if a cached result exists.",
    )
    parser.add_argument(
        "--verification-protocol",
        choices=["tar_at_far", "eer"],
        default="tar_at_far",
        help="Protocol used to fit the verification decision threshold (default: tar_at_far). "
        "EER is always computed and reported regardless of this choice.",
    )
    parser.add_argument(
        "--target-far",
        type=float,
        default=1e-3,
        help="Target False Acceptance Rate for the tar_at_far protocol (default: 1e-3).",
    )
    parser.add_argument("--random-seed", type=int, default=1337)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "output" / "evaluations",
        help="Root directory where evaluation artefacts are written",
    )
    return parser


def _load_inception_model(device: torch.device):
    from torchvision.models import inception_v3, Inception_V3_Weights

    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # type: ignore[assignment]
    model.eval().to(device)
    return model


def _extract_inception_features(
    images: Dict[str, np.ndarray],
    model,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    import cv2 as _cv2
    from torchvision.transforms.functional import resize

    all_feats: List[np.ndarray] = []
    imgs = list(images.values())
    for i in range(0, len(imgs), batch_size):
        tensors = []
        for bgr in imgs[i : i + batch_size]:
            rgb = _cv2.cvtColor(bgr, _cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            t = resize(t, [299, 299], antialias=True)
            t = (t - 0.5) / 0.5
            tensors.append(t)
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            feats = model(batch)
        all_feats.append(feats.cpu().numpy())
    return np.concatenate(all_feats, axis=0)


def _fid_from_features(real_feats: np.ndarray, anon_feats: np.ndarray) -> float:
    from scipy.linalg import sqrtm

    eps = 1e-6
    mu_r, mu_a = real_feats.mean(0), anon_feats.mean(0)
    sigma_r = np.cov(real_feats, rowvar=False) + eps * np.eye(real_feats.shape[1])
    sigma_a = np.cov(anon_feats, rowvar=False) + eps * np.eye(anon_feats.shape[1])
    diff = mu_r - mu_a
    covmean: np.ndarray = sqrtm(sigma_r @ sigma_a)  # type: ignore[assignment]
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # type: ignore[assignment]
    return float(diff @ diff + np.trace(sigma_r + sigma_a - 2.0 * covmean))


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
    variant = None
    if args.embedder == "adaface":
        variant = args.adaface_model_type           # preserve existing CLI knob
        cache_suffix = f"adaface_{args.adaface_model_type}"
    else:
        cache_suffix = args.embedder
    embedder = get_eval_embedder(
        args.embedder, device, batch_size=args.batch_size, variant=variant
    )

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
            device=str(device),
        )
        df = evaluator.run()
        if args.label:
            out_path = args.output_dir / args.label / "utility.csv"
        else:
            out_path = args.output_dir / result_dataset_name / label / "utility.csv"
        ensure_csv_parent(out_path)
        df.to_csv(out_path, index=False)
        report_utility(df)

        # FID: compare three pairs to separate reconstruction error from masking error
        _GT_COUNTERPART = {
            "CombinedReconst": "CombinedGT",
            "NeRSembleReconst": "NeRSembleGT",
            "FaceScapeReconst": "FaceScapeGT",
        }
        gt_dataset_name = _GT_COUNTERPART.get(anonymized_dataset_name)
        gt_images: Optional[Dict[str, np.ndarray]] = None
        if gt_dataset_name is not None:
            try:
                gt_spec = resolve_dataset(gt_dataset_name)  # type: ignore[arg-type]
                gt_dataset = FaceDataset(gt_spec.images_root, gt_spec.file_extension)
                gt_images, _ = load_image_map(
                    [str(p.relative_to(gt_spec.images_root)) for p in gt_dataset.paths],
                    gt_spec.images_root,
                    key_prefix=gt_spec.name,
                )
            except (FileNotFoundError, RuntimeError):
                print(f"[FID] GT dataset '{gt_dataset_name}' not found, skipping GT-based FID.")

        inception = _load_inception_model(device)
        unmasked_feats = _extract_inception_features(real_images, inception, device, args.batch_size)
        masked_feats = _extract_inception_features(anon_images, inception, device, args.batch_size)

        print("================ FID Results ================")
        if gt_images:
            gt_feats = _extract_inception_features(gt_images, inception, device, args.batch_size)
            print(f"FID (GT vs unmasked): {_fid_from_features(gt_feats, unmasked_feats):.4f}")
            print(f"FID (GT vs masked):   {_fid_from_features(gt_feats, masked_feats):.4f}")
        print(f"FID (unmasked vs masked): {_fid_from_features(unmasked_feats, masked_feats):.4f}")
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
            protocol=args.verification_protocol,
            target_far=args.target_far,
        )

        # Similarity-space decision threshold selected by the chosen protocol.
        decision_score_threshold = threshold_result.score_threshold
        if threshold_result.protocol == "tar_at_far":
            print(
                f"Using TAR@FAR threshold {decision_score_threshold:.4f} "
                f"(target FAR {threshold_result.target_far:.0e}; "
                f"TAR {threshold_result.tar_at_far:.2%}, FAR {threshold_result.far_at_far:.2%}, "
                f"FRR {threshold_result.frr_at_far:.2%}) derived from {threshold_dataset_name} "
                f"(EER {threshold_result.eer:.2%}; EER thr {threshold_result.eer_score_threshold:.4f}; "
                f"cache: {threshold_result.cache_file})."
            )
        else:
            print(
                f"Using EER threshold {decision_score_threshold:.4f} derived from {threshold_dataset_name} "
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
        # Primary decision threshold follows the selected protocol (default
        # tar_at_far). The EER threshold is kept available for the match_eer
        # column so the EER protocol stays reported.
        threshold = decision_score_threshold
        eer_threshold = threshold_result.eer_score_threshold
        tar_far_threshold = threshold_result.tar_at_far_threshold

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
                min_similarity = similarity.min().item()
                pred = 1 if max_similarity >= threshold else 0

                rows.append(
                    {
                        "subject_id": anon_id,
                        "emb_idx": anon_emb_idx,
                        # `match` follows the selected protocol's threshold.
                        "match": True if pred == 1 else False,
                        "max_similarity": max_similarity,
                        "min_similarity": min_similarity,
                        # Per-protocol decisions kept side by side so both the
                        # TAR@FAR and EER operating points stay reported.
                        "match_eer": bool(max_similarity >= eer_threshold),
                        "match_tar_at_far": bool(max_similarity >= tar_far_threshold),
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
            protocol=threshold_result.protocol,
            target_far=threshold_result.target_far,
            tar_at_far=threshold_result.tar_at_far,
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
        "fid",
        "age_diff",
        "emotion_match",
        "gender_match",
        "race_match",
    ]:
        if metric in df.columns:
            value = df[metric].mean()
            if metric == "age_diff":
                print(f"Average Age Difference: {value:.2f} years")
            elif metric == "fid":
                # Set-level scalar broadcast across all rows (mean == the value).
                print(f"FID (unmasked vs masked, set-level): {value:.4f}")
            elif metric in ["ssim", "psnr"]:
                std = df[metric].std()
                print(f"Average {metric.upper()}: {value:.4f} ± {std:.4f}")
            else:
                print(f"Average {metric.replace('_', ' ').title()}: {value:.2%}")


def report_verification(
    df: pd.DataFrame,
    threshold: float,
    fitted_eer: Optional[float] = None,
    fitted_eer_score: Optional[float] = None,
    protocol: str = "eer",
    target_far: float = 0.0,
    tar_at_far: float = 0.0,
) -> None:
    if df.empty:
        print("No verification pairs available")
        return
    accuracy = df["match"].mean()
    print("================ Verification Results ================")
    print(f"Protocol: {protocol}")
    print(f"Threshold (fitted on real pairs): {threshold:.4f}")
    if protocol == "tar_at_far":
        print(f"Target FAR: {target_far:.0e}  (gallery TAR@FAR: {tar_at_far:.2%})")
    if fitted_eer is not None:
        print(f"Gallery EER (real pairs): {fitted_eer:.2%}")
    if fitted_eer_score is not None:
        print(f"EER score threshold (cosine similarity): {fitted_eer_score:.4f}")
    print(f"Match accuracy (on anonymized dataset): {accuracy:.2%}")
    if "match_eer" in df.columns:
        print(f"Match accuracy @ EER threshold: {df['match_eer'].mean():.2%}")
    if "match_tar_at_far" in df.columns:
        print(f"Match accuracy @ TAR@FAR threshold: {df['match_tar_at_far'].mean():.2%}")


if __name__ == "__main__":
    main()
