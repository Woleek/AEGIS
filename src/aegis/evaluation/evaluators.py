from collections import defaultdict
from dataclasses import dataclass
import cv2
from deepface import DeepFace
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from pathlib import Path

from sklearn.metrics import roc_curve
from .stores import ThresholdCacheStore, VerificationThresholdResult
from ..utils import plot_far_frr_with_eer
from .datasets import DatasetIdentityLookup


import numpy as np
import pandas as pd
from tqdm import tqdm


from typing import Dict, List, Literal, Optional, Sequence, Tuple

from ..models.base import load_insightface_detector


# Default operating point for the TAR@FAR verification protocol.
DEFAULT_TARGET_FAR = 1e-3


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(vec1, vec2) / denom)


@dataclass
class ThresholdResult:
    """Operating point of a verification threshold, in cosine-similarity space.

    ``threshold`` is a *similarity* lower bound: a pair verifies when its cosine
    similarity is ``>= threshold`` (higher = more similar).
    """

    name: str
    threshold: float
    tar: float
    far: float
    frr: float  # false rejection rate = 1 - TAR
    accuracy: float


def _sorted_thresholds(
    pos_sims: Sequence[float], neg_sims: Sequence[float]
) -> np.ndarray:
    """All unique similarity values as candidate thresholds, descending."""
    return np.sort(np.unique(np.concatenate([pos_sims, neg_sims])))[::-1]


def compute_tar_at_far(
    pos_sims: Sequence[float],
    neg_sims: Sequence[float],
    target_far: float = DEFAULT_TARGET_FAR,
) -> ThresholdResult:
    """Lowest similarity threshold (highest TAR) where FAR <= target_far."""
    pos = np.asarray(pos_sims, dtype=np.float64)
    neg = np.asarray(neg_sims, dtype=np.float64)

    # Fallback: most conservative threshold (guaranteed FAR=0)
    thresholds = _sorted_thresholds(pos_sims, neg_sims)  # descending
    threshold = float(thresholds[0])
    tar = float(np.mean(pos >= threshold))
    far = float(np.mean(neg >= threshold))

    for t in thresholds:
        f = float(np.mean(neg >= t))
        if f <= target_far:
            # Still within budget — go lower to improve TAR
            threshold, tar, far = float(t), float(np.mean(pos >= t)), f
        else:
            # FAR constraint violated — stop
            break

    frr = 1.0 - tar
    accuracy = float(
        (np.sum(pos >= threshold) + np.sum(neg < threshold)) / (len(pos) + len(neg))
    )
    return ThresholdResult("TAR@FAR", threshold, tar, far, frr, accuracy)


def compute_eer_similarity(
    pos_sims: Sequence[float],
    neg_sims: Sequence[float],
) -> ThresholdResult:
    """Similarity threshold where FAR ~= FRR (Equal Error Rate)."""
    pos = np.asarray(pos_sims, dtype=np.float64)
    neg = np.asarray(neg_sims, dtype=np.float64)

    best_threshold = 0.0
    best_diff = float("inf")

    for t in _sorted_thresholds(pos_sims, neg_sims):
        far = float(np.mean(neg >= t))
        frr = float(np.mean(pos < t))
        diff = abs(far - frr)
        if diff < best_diff:
            best_diff = diff
            best_threshold = float(t)

    tar = float(np.mean(pos >= best_threshold))
    far = float(np.mean(neg >= best_threshold))
    frr = 1.0 - tar
    accuracy = float(
        (np.sum(pos >= best_threshold) + np.sum(neg < best_threshold))
        / (len(pos) + len(neg))
    )
    return ThresholdResult("EER", best_threshold, tar, far, frr, accuracy)


class RankKEvaluator:
    def __init__(
        self,
        identity_lookup: DatasetIdentityLookup,
        real_embeddings: Dict[str, np.ndarray],
        anon_embeddings: Dict[str, np.ndarray],
        limit_one_per_identity: bool,
    ) -> None:
        self.identity_lookup = identity_lookup
        self.real_embeddings = real_embeddings
        self.anon_embeddings = anon_embeddings
        self.limit_one_per_identity = limit_one_per_identity

    def run(self) -> pd.DataFrame:
        reference_items = list(self.real_embeddings.items())
        if self.limit_one_per_identity:
            reference_items = self._one_per_identity(reference_items)

        ref_keys = [key for key, _ in reference_items]
        ref_matrix = np.vstack([vec for _, vec in reference_items])

        results: List[Dict[str, float]] = []
        for query_key, query_vec in tqdm(
            self.anon_embeddings.items(), desc="Evaluating rank-k"
        ):
            try:
                query_id = self.identity_lookup.lookup(query_key)
            except Exception:
                print("Could not resolve identity for query key, skipping.")
                continue
            similarities = 1.0 - np.dot(ref_matrix, query_vec)
            order = np.argsort(similarities)
            for k_offset, rank in enumerate(order):
                candidate_key = ref_keys[rank]
                # print(
                #     f"Comparing query '{query_key}' to candidate '{candidate_key}' at rank {k_offset + 1}"
                # )
                try:
                    candidate_id = self.identity_lookup.lookup(candidate_key)

                    if candidate_id == query_id:
                        results.append(
                            {
                                "query_key": query_key,
                                "k": int(k_offset),
                                "distance": float(similarities[rank]),
                            }
                        )
                        break
                except Exception:
                    # Could not resolve identity for candidate (different dataset)
                    pass

                if k_offset >= 50:
                    results.append(
                        {
                            "query_key": query_key,
                            "k": 999,
                            "distance": float(similarities[rank]),
                        }
                    )
                    break
        return pd.DataFrame(results)

    def _one_per_identity(
        self, reference_items: List[Tuple[str, np.ndarray]]
    ) -> List[Tuple[str, np.ndarray]]:
        seen: Dict[str, Tuple[str, np.ndarray]] = {}
        for key, vec in reference_items:
            try:
                ident = self.identity_lookup.lookup(key)
            except Exception:
                continue
            if ident not in seen:
                seen[ident] = (key, vec)
        return list(seen.values())


class UtilityEvaluator:
    def __init__(
        self,
        identity_lookup: DatasetIdentityLookup,
        real_images: Dict[str, np.ndarray],
        real_paths: List[Path],
        anon_images: Dict[str, np.ndarray],
        anon_paths: List[Path],
        device: Optional[str] = None,
        fid_batch_size: int = 8,
    ) -> None:
        self.detection_model = load_insightface_detector(ctx_id=0)
        self.identity_lookup = identity_lookup
        self.real_images = real_images
        self.anon_images = anon_images
        self.real_paths = real_paths
        self.anon_paths = anon_paths
        self.device = device
        self.fid_batch_size = max(1, int(fid_batch_size))

        shared_keys = set(real_images.keys()) & set(anon_images.keys())
        missing_in_anon = set(real_images.keys()) - shared_keys
        missing_in_real = set(anon_images.keys()) - shared_keys

        if missing_in_anon:
            sample = ", ".join(sorted(list(missing_in_anon))[:5])
            print(
                f"[utility] {len(missing_in_anon)} real images were missing anonymized counterparts. "
                f"Examples: {sample}"
            )
        if missing_in_real:
            sample = ", ".join(sorted(list(missing_in_real))[:5])
            print(
                f"[utility] {len(missing_in_real)} anonymized images had no real counterpart. "
                f"Examples: {sample}"
            )

        if not shared_keys:
            raise ValueError(
                "No overlapping samples between real and anonymized images."
            )

        self.paired_keys: List[str] = sorted(shared_keys)
        self.paired_paths: List[Tuple[Path, Path]] = []
        for real_path in self.real_paths:
            key = real_path.name
            anon_path = next((p for p in self.anon_paths if p.name == key), None)
            if anon_path is not None:
                self.paired_paths.append((real_path, anon_path))

    def run(self) -> pd.DataFrame:
        # Utility evaluation using DeepFace codebase, VGG-Face (calculate between unaltered and masked face):
        # - Structural Similarity Index Measure (SSIM)
        # - Peak Signal-to-Noise Ratio (PSNR)
        # - Emotion Classification
        # - Gender Classification
        # - Race Classification
        # - Age Prediction

        # Build path lookup keyed by bare filename (e.g. "074.png")
        path_by_filename: Dict[str, Tuple[Path, Path]] = {}
        for real_path, anon_path in self.paired_paths:
            path_by_filename[real_path.name] = (real_path, anon_path)

        ssim_scores = self._calculate_ssim()
        psnr_scores = self._calculate_psnr()
        # Set-level scalar: same FID value is broadcast into every row so the
        # 28-row per-subject schema of utility.csv is preserved.
        fid_value = self._calculate_fid()
        attr_by_filename = self._measure_utility(path_by_filename)

        rows = []
        for key, ssim_val, psnr_val in zip(self.paired_keys, ssim_scores, psnr_scores):
            # Extract bare filename: "CombinedReconst___074.png" → "074.png"
            filename = key.split("___")[-1] if "___" in key else key
            subject_id = Path(filename).stem
            em, gm, rm, ad = attr_by_filename.get(filename, (float("nan"),) * 4)
            rows.append(
                {
                    "subject_id": subject_id,
                    "ssim": ssim_val,
                    "psnr": psnr_val,
                    "fid": fid_value,
                    "emotion_match": em,
                    "gender_match": gm,
                    "race_match": rm,
                    "age_diff": ad,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _images_to_fid_batch(images: Dict[str, np.ndarray]):
        """Convert a dict of BGR uint8 numpy images into an (N, 3, H, W) float
        tensor in [0, 1] (RGB, CHW) suitable for ``FrechetInceptionDistance``
        constructed with ``normalize=True``.

        Uses the same BGR->RGB conversion as the SSIM/PSNR methods. All images
        are resized to a common (largest) spatial size so they can be batched;
        Inception internally resizes to 299x299, so this only needs to yield a
        stackable batch.
        """
        import torch

        rgb_arrays: List[np.ndarray] = []
        max_h = 0
        max_w = 0
        for img in images.values():
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            rgb_arrays.append(rgb)
            max_h = max(max_h, rgb.shape[0])
            max_w = max(max_w, rgb.shape[1])

        tensors = []
        for rgb in rgb_arrays:
            if rgb.shape[0] != max_h or rgb.shape[1] != max_w:
                rgb = cv2.resize(rgb, (max_w, max_h), interpolation=cv2.INTER_AREA)
            # HWC uint8 [0,255] -> CHW float [0,1]
            t = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)
            tensors.append(t.float() / 255.0)

        return torch.stack(tensors)

    @staticmethod
    def _build_fid_metric():
        """Construct a ``FrechetInceptionDistance(feature=2048, normalize=True)``.

        The canonical integer form ``feature=2048`` relies on ``torch-fidelity``
        for its Inception-V3 feature extractor. When that optional dependency is
        not installed, torchmetrics still supports passing a custom ``nn.Module``
        feature extractor; we then supply a torchvision Inception-V3 (pool3,
        2048-dim) wrapper, which keeps the metric identical in spirit (same
        2048-dim Inception pool features, [0,1] float inputs, internal resize)
        without adding any new dependency.
        """
        import torch
        from torchmetrics.image.fid import FrechetInceptionDistance

        try:
            return FrechetInceptionDistance(feature=2048, normalize=True)
        except ModuleNotFoundError:
            pass

        from torchvision.models import inception_v3, Inception_V3_Weights
        import torch.nn.functional as F

        class _TorchvisionInception2048(torch.nn.Module):
            """torchvision Inception-V3 truncated at the 2048-dim pool features.

            Accepts (N, 3, H, W) float images in [0, 1] (torchmetrics passes
            these because the metric was created with ``normalize=True``) and
            returns (N, 2048) features. Resizing to 299x299 and ImageNet
            normalization are handled internally, mirroring the standard
            FID feature extractor contract.
            """

            num_features = 2048

            def __init__(self) -> None:
                super().__init__()
                self.model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
                self.model.fc = torch.nn.Identity()
                self.model.eval()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
                self.register_buffer("_mean", mean)
                self.register_buffer("_std", std)

            @torch.no_grad()
            def forward(self, x: torch.Tensor) -> torch.Tensor:
                x = x.float()
                if x.max() > 1.5:  # tolerate uint8-style [0,255] inputs
                    x = x / 255.0
                x = F.interpolate(
                    x, size=(299, 299), mode="bilinear", align_corners=False
                )
                x = (x - self._mean) / self._std
                return self.model(x)

        return FrechetInceptionDistance(
            feature=_TorchvisionInception2048(), normalize=True
        )

    def _calculate_fid(self) -> float:
        """Compute a single set-level Frechet Inception Distance between the
        full set of real (unmasked) images and the full set of anonymized
        (masked) images.

        Returns NaN if FID cannot be computed (e.g. Inception weights are not
        available offline).
        """
        real_batch = self._images_to_fid_batch(self.real_images)
        anon_batch = self._images_to_fid_batch(self.anon_images)

        n_real = real_batch.shape[0]
        n_anon = anon_batch.shape[0]
        n_min = min(n_real, n_anon)
        print(f"[FID] Computing FID over {n_real} real and {n_anon} anon images.")
        if n_min < 50:
            print(
                "[FID] WARNING: FID is being computed from a small number of "
                f"samples (min set size = {n_min} < 50). FID estimates with so "
                "few images are high-variance and unreliable; treat the value "
                "as indicative only."
            )

        device = self.device or "cpu"
        bs = self.fid_batch_size
        try:
            import torch

            fid = self._build_fid_metric().to(device)
            for batch, is_real in ((real_batch, True), (anon_batch, False)):
                for i in range(0, batch.shape[0], bs):
                    chunk = batch[i : i + bs].to(device)
                    fid.update(chunk, real=is_real)
                    del chunk
                    if device != "cpu":
                        torch.cuda.empty_cache()
            value = float(fid.compute().item())
        except Exception as exc:  # noqa: BLE001
            print(
                f"[FID] WARNING: failed to compute FID ({type(exc).__name__}: {exc}). "
                "Returning NaN. This commonly happens when Inception weights "
                "cannot be downloaded offline."
            )
            return float("nan")

        return value

    def _calculate_ssim(self) -> List[float]:
        scores: List[float] = []
        for key in tqdm(self.paired_keys, desc="Calculating SSIM"):
            real_img = self.real_images[key]
            anon_img = self.anon_images[key]

            real_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            anon_rgb = cv2.cvtColor(anon_img, cv2.COLOR_BGR2RGB)

            if anon_rgb.shape[:2] != real_rgb.shape[:2]:
                anon_rgb = cv2.resize(
                    anon_rgb, (real_rgb.shape[1], real_rgb.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            bboxes, _ = self.detection_model.detect(real_rgb, max_num=1)

            if bboxes is not None and len(bboxes) > 0:
                x1, y1, x2, y2, _ = bboxes[0].astype(int)
                real_rgb = real_rgb[y1:y2, x1:x2]
                anon_rgb = anon_rgb[y1:y2, x1:x2]

            ssim_value = structural_similarity(
                real_rgb, anon_rgb, data_range=255.0, channel_axis=2
            )
            scores.append(float(ssim_value))

        return scores

    def _calculate_psnr(self) -> List[float]:
        scores: List[float] = []
        for key in tqdm(self.paired_keys, desc="Calculating PSNR"):
            real_img = self.real_images[key]
            anon_img = self.anon_images[key]

            real_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            anon_rgb = cv2.cvtColor(anon_img, cv2.COLOR_BGR2RGB)

            if anon_rgb.shape[:2] != real_rgb.shape[:2]:
                anon_rgb = cv2.resize(
                    anon_rgb, (real_rgb.shape[1], real_rgb.shape[0]),
                    interpolation=cv2.INTER_CUBIC,
                )

            bboxes, _ = self.detection_model.detect(real_rgb, max_num=1)

            if bboxes is not None and len(bboxes) > 0:
                x1, y1, x2, y2, _ = bboxes[0].astype(int)
                real_rgb = real_rgb[y1:y2, x1:x2]
                anon_rgb = anon_rgb[y1:y2, x1:x2]

            psnr_value = peak_signal_noise_ratio(real_rgb, anon_rgb, data_range=255)
            scores.append(float(psnr_value))

        return scores

    def _measure_utility(
        self, path_by_filename: Dict[str, Tuple[Path, Path]]
    ) -> Dict[str, Tuple]:
        """Returns results keyed by bare filename (e.g. '074.png')."""
        results: Dict[str, Tuple] = {}

        for key in tqdm(self.paired_keys, desc="Measuring utility metrics"):
            filename = key.split("___")[-1] if "___" in key else key
            pair = path_by_filename.get(filename)
            if pair is None:
                continue
            real_path, anon_path = pair
            try:
                real_analysis = DeepFace.analyze(
                    img_path=str(real_path),
                    actions=["emotion", "gender", "race", "age"],
                    enforce_detection=False,
                )
                anon_analysis = DeepFace.analyze(
                    img_path=str(anon_path),
                    actions=["emotion", "gender", "race", "age"],
                    enforce_detection=False,
                )
            except Exception as e:
                print(f"Error analyzing images {real_path} and {anon_path}: {e}")
                continue

            emotion_match = (
                real_analysis[0]["dominant_emotion"]
                == anon_analysis[0]["dominant_emotion"]
            )
            gender_match = (
                real_analysis[0]["dominant_gender"]
                == anon_analysis[0]["dominant_gender"]
            )
            race_match = (
                real_analysis[0]["dominant_race"] == anon_analysis[0]["dominant_race"]
            )
            age_diff = abs(real_analysis[0]["age"] - anon_analysis[0]["age"])

            results[filename] = (
                float(emotion_match),
                float(gender_match),
                float(race_match),
                float(age_diff),
            )

        return results


class PairVerificationEvaluator:
    def __init__(
        self,
        identity_lookup: DatasetIdentityLookup,
        real_embeddings: Dict[str, np.ndarray],
        anon_embeddings: Dict[str, np.ndarray],
        num_pairs: int,
        random_seed: int,
    ) -> None:
        self.identity_lookup = identity_lookup
        self.real_embeddings = real_embeddings
        self.anon_embeddings = anon_embeddings
        self.num_pairs = num_pairs
        self.random = np.random.default_rng(random_seed)

        # Efficient pair lookup by identity
        self.identity_to_keys: Dict[str, List[str]] = defaultdict(list)
        for key in tqdm(
            self.real_embeddings.keys(), desc="Building identity to keys mapping"
        ):
            try:
                identity = self.identity_lookup.lookup(key)
                self.identity_to_keys[identity].append(key)
            except Exception:
                print(f"Could not resolve identity for key '{key}', skipping.")
                continue

        # Filter identities:
        # - all_identities: Used for picking negative pairs
        # - identities_with_pairs: Used for picking positive pairs (need >= 2 samples)
        self.all_identities = list(self.identity_to_keys.keys())
        self.identities_with_pairs = [
            identity
            for identity, keys in self.identity_to_keys.items()
            if len(keys) >= 2
        ]

        if not self.all_identities:
            raise ValueError("No valid identities found in real_embeddings.")
        if not self.identities_with_pairs:
            raise ValueError(
                "No identities with 2 or more samples found, "
                "cannot generate positive pairs."
            )

    def run(
        self,
        threshold: Optional[float] = None,
        protocol: Literal["tar_at_far", "eer"] = "tar_at_far",
        target_far: float = DEFAULT_TARGET_FAR,
    ) -> Tuple[pd.DataFrame, float, float, float]:
        """Fit a verification threshold on the real pairs and predict anon pairs.

        ``protocol`` selects how the (distance) threshold is fit:
          - ``"tar_at_far"`` (default): operating point with FAR <= ``target_far``.
          - ``"eer"``: legacy Equal Error Rate operating point.

        The EER is always computed and returned regardless of protocol so callers
        keep reporting it. Returns ``(df, threshold, eer, eer_threshold)`` where
        ``threshold`` and ``eer_threshold`` are DISTANCE thresholds (1 - cosine).
        The chosen TAR@FAR / EER similarity-space operating points are stashed on
        ``self.last_threshold_result`` / ``self.last_eer_result`` for reporting.
        """
        real_pairs, anon_pairs = self._build_pairs()
        if threshold is None:
            threshold = self._fit_threshold(
                real_pairs, protocol=protocol, target_far=target_far
            )
        df = self._predict(anon_pairs, threshold)
        eer, eer_threshold = self._compute_eer(df)
        return df, threshold, eer, eer_threshold

    def _build_pairs(self) -> Tuple[List[Tuple], List[Tuple]]:
        real_pairs: List[Tuple] = []
        anon_pairs: List[Tuple] = []

        # 1. Positive pairs
        while len(real_pairs) < self.num_pairs:
            # Pick a random identity that has at least 2 samples
            identity = self.random.choice(self.identities_with_pairs)

            # Pick two different keys from that identity
            key_a, key_b = self.random.choice(
                self.identity_to_keys[identity], size=2, replace=False
            )

            emb_a_real = self.real_embeddings[key_a]
            emb_b_real = self.real_embeddings[key_b]
            real_pairs.append((key_a, key_b, emb_a_real, emb_b_real, 1))

            # Check for corresponding anon embedding for key_b
            emb_b_anon = self.anon_embeddings.get(key_b)
            if emb_b_anon is not None:
                anon_pairs.append((key_a, key_b, emb_a_real, emb_b_anon, 1))

            # update the bar in a while loop
            if len(real_pairs) == 1:
                pbar_pos = tqdm(total=self.num_pairs, desc="Building positive pairs")
                pbar_pos.update(1)
            elif "pbar_pos" in locals():
                pbar_pos.update(1)

        if "pbar_pos" in locals():
            pbar_pos.close()

        # 2. Negative pairs
        for _ in tqdm(range(self.num_pairs), desc="Building negative pairs"):
            # Pick two different random identities
            identity_a, identity_b = self.random.choice(
                self.all_identities, size=2, replace=False
            )

            # Pick one key from each
            key_a = self.random.choice(self.identity_to_keys[identity_a])
            key_b = self.random.choice(self.identity_to_keys[identity_b])

            emb_a_real = self.real_embeddings[key_a]
            emb_b_real = self.real_embeddings[key_b]
            real_pairs.append((key_a, key_b, emb_a_real, emb_b_real, 0))

            # Check for corresponding anon embedding for key_b
            emb_b_anon = self.anon_embeddings.get(key_b)
            if emb_b_anon is not None:
                anon_pairs.append((key_a, key_b, emb_a_real, emb_b_anon, 0))

        return real_pairs, anon_pairs

    def _fit_threshold(
        self,
        pairs: Sequence[Tuple],
        protocol: Literal["tar_at_far", "eer"] = "tar_at_far",
        target_far: float = DEFAULT_TARGET_FAR,
    ) -> float:
        distances: List[float] = []
        labels: List[int] = []
        for _, _, emb_a, emb_b, label in tqdm(pairs, desc="Fitting distances"):
            distances.append(cosine_distance(emb_a, emb_b))
            labels.append(label)

        if not distances:
            raise ValueError("No pairs provided to fit threshold.")

        distances_arr = np.asarray(distances, dtype=np.float32)
        labels_arr = np.asarray(labels, dtype=np.int8)
        if np.unique(labels_arr).size < 2:
            raise ValueError("Need both positive and negative pairs to fit threshold.")

        # This evaluator works in DISTANCE
        # space (distance = 1 - cosine; a pair matches when distance < threshold).
        # We therefore fit in similarity space, then convert the chosen
        # similarity threshold back to a distance threshold via 1 - similarity so
        # it integrates with `_predict` unchanged.
        scores = 1.0 - distances_arr  # cosine similarity
        pos_sims = scores[labels_arr == 1].tolist()
        neg_sims = scores[labels_arr == 0].tolist()

        # Always compute EER (similarity space) so it can be reported regardless
        # of which protocol drives the decision threshold.
        eer_result = compute_eer_similarity(pos_sims, neg_sims)
        self.last_eer_result = eer_result
        self._last_fit_eer = float((eer_result.far + eer_result.frr) / 2.0)
        self._last_fit_eer_score_threshold = float(eer_result.threshold)

        if protocol == "tar_at_far":
            chosen = compute_tar_at_far(pos_sims, neg_sims, target_far)
        elif protocol == "eer":
            chosen = eer_result
        else:
            raise ValueError(f"Unknown verification protocol '{protocol}'.")

        self.last_threshold_result = chosen
        self.last_protocol = protocol
        self.last_target_far = float(target_far)

        distance_threshold = float(1.0 - chosen.threshold)
        return distance_threshold

    def _predict(self, pairs: Sequence[Tuple], threshold: float) -> pd.DataFrame:
        rows = []
        for key_a, key_b, emb_a, emb_b, label in tqdm(
            pairs, desc="Predicting distances"
        ):
            distance = cosine_distance(emb_a, emb_b)
            pred = 1 if distance < threshold else 0
            rows.append(
                {
                    "key_a": key_a,
                    "key_b": key_b,
                    "label": label,
                    "pred": pred,
                    "distance": distance,
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def _compute_eer_from_scores(
        y_true: np.ndarray, scores: np.ndarray
    ) -> Tuple[float, float]:
        """
        Calculates the EER and EER threshold score using interpolation.

        Returns:
            Tuple[float, float]: (eer, eer_threshold_score)
        """
        if np.unique(y_true).size < 2:
            return 0.5, 0.0  # Not enough labels to compute EER

        fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)

        # Filter out non-finite thresholds
        valid = ~np.isinf(thresholds)
        if not np.any(valid):
            # No valid finite thresholds. Find closest point on unfiltered data.
            frr_all = 1 - tpr
            idx = np.nanargmin(np.abs(frr_all - fpr))
            # Note: thresholds[idx] might be inf
            eer = (frr_all[idx] + fpr[idx]) / 2.0
            return float(eer), float(thresholds[idx])

        # Use only finite thresholds for interpolation
        fpr, tpr, thresholds = fpr[valid], tpr[valid], thresholds[valid]

        frr = 1.0 - tpr
        diff = frr - fpr
        # Find the point where frr crosses fpr
        crossing = np.where(np.sign(diff[1:]) != np.sign(diff[:-1]))[0]

        if crossing.size == 0:
            # No crossing, find the closest point
            idx = int(np.nanargmin(np.abs(diff)))
            eer = (fpr[idx] + frr[idx]) / 2.0
            threshold_score = float(thresholds[idx])
        else:
            # Interpolate to find the exact crossing point
            j = int(crossing[0])
            x0, y0, t0 = fpr[j], frr[j], thresholds[j]
            x1, y1, t1 = fpr[j + 1], frr[j + 1], thresholds[j + 1]

            denom = (y1 - y0) - (x1 - x0)
            if abs(denom) < 1e-12:
                t = 0.0
            else:
                t = (x0 - y0) / denom

            t = float(np.clip(t, 0.0, 1.0))
            eer = x0 + t * (x1 - x0)
            threshold_score = float(t0 + t * (t1 - t0))

        return float(eer), float(threshold_score)

    @staticmethod
    def _compute_eer(df: pd.DataFrame) -> Tuple[float, float]:
        if df.empty:
            return 0.5, 0.0  # (EER, EER threshold)

        y_true = df["label"].to_numpy()
        scores = df["similarity"].to_numpy()

        # Call the new shared static method
        eer, eer_threshold_score = PairVerificationEvaluator._compute_eer_from_scores(
            y_true, scores
        )

        return eer, eer_threshold_score


# Backward-/forward-compatible alias. The pair-based 1:1 evaluator is the
# project's verification evaluator; expose it under the generic name too.
VerificationEvaluator = PairVerificationEvaluator


def compute_or_load_verification_threshold(
    dataset_name: str,
    embedder_key: str,
    embeddings: Dict[str, np.ndarray],
    identity_lookup: DatasetIdentityLookup,
    thresholds_root: Path,
    num_pairs: int,
    random_seed: int,
    force_recompute: bool = False,
    protocol: Literal["tar_at_far", "eer"] = "tar_at_far",
    target_far: float = DEFAULT_TARGET_FAR,
) -> VerificationThresholdResult:
    if not embeddings:
        raise ValueError(
            f"No embeddings available for threshold dataset '{dataset_name}'."
        )

    threshold_dir = thresholds_root / dataset_name
    threshold_dir.mkdir(parents=True, exist_ok=True)
    cache_file = threshold_dir / f"{dataset_name}_{embedder_key}.json"
    plot_file = threshold_dir / f"{dataset_name}_{embedder_key}_far_frr.png"

    cache_store = ThresholdCacheStore(cache_file)
    if not force_recompute:
        cached = cache_store.load()
        # Only reuse a cache that matches the requested protocol/target FAR; older
        # caches predate TAR@FAR (no "protocol" key) and must be recomputed when
        # TAR@FAR is requested.
        cached_protocol = (cached or {}).get("protocol", "eer")
        cached_target_far = float((cached or {}).get("target_far", 0.0))
        compatible = (
            cached_protocol == protocol
            and (protocol != "tar_at_far" or cached_target_far == float(target_far))
        )
        if (
            cached
            and cached.get("dataset") == dataset_name
            and cached.get("embedder") == embedder_key
            and compatible
        ):
            plot_path = Path(cached.get("plot_path", plot_file))
            if not plot_path.is_absolute():
                plot_path = threshold_dir / plot_path
            distance_threshold = float(cached["distance_threshold"])
            return VerificationThresholdResult(
                dataset=cached.get("dataset", dataset_name),
                embedder=cached.get("embedder", embedder_key),
                distance_threshold=distance_threshold,
                eer=float(cached["eer"]),
                eer_score_threshold=float(cached["eer_score_threshold"]),
                far=float(cached.get("far", 0.0)),
                frr=float(cached.get("frr", 0.0)),
                roc_auc=float(cached.get("roc_auc", 0.0)),
                cache_file=cache_file,
                plot_path=plot_path,
                protocol=cached_protocol,
                score_threshold=float(
                    cached.get("score_threshold", 1.0 - distance_threshold)
                ),
                target_far=cached_target_far,
                tar_at_far_threshold=float(cached.get("tar_at_far_threshold", 0.0)),
                tar_at_far=float(cached.get("tar_at_far", 0.0)),
                far_at_far=float(cached.get("far_at_far", 0.0)),
                frr_at_far=float(cached.get("frr_at_far", 0.0)),
            )

    pv = PairVerificationEvaluator(
        identity_lookup=identity_lookup,
        real_embeddings=embeddings,
        anon_embeddings=embeddings,
        num_pairs=num_pairs,
        random_seed=random_seed,
    )
    real_pairs, _ = pv._build_pairs()
    if not real_pairs:
        raise RuntimeError(
            f"Failed to construct real pairs for threshold dataset '{dataset_name}'."
        )

    pos_scores: List[float] = []
    neg_scores: List[float] = []
    for _, _, emb_a, emb_b, label in real_pairs:
        score = 1.0 - cosine_distance(emb_a, emb_b)
        if label == 1:
            pos_scores.append(score)
        else:
            neg_scores.append(score)

    pos_scores_arr = np.asarray(pos_scores, dtype=np.float32)
    neg_scores_arr = np.asarray(neg_scores, dtype=np.float32)

    # EER (similarity space) — always computed and reported via the FAR/FRR plot.
    thr_eer, eer, far_at_thr, frr_at_thr, roc_auc = plot_far_frr_with_eer(
        pos_scores_arr,
        neg_scores_arr,
        title=f"{dataset_name} verification FAR/FRR",
        save_path=plot_file,
    )

    # TAR@FAR operating point (similarity space). This is the DEFAULT decision
    # protocol: it picks the lowest similarity threshold whose FAR <= target_far.
    tar_far_result = compute_tar_at_far(pos_scores, neg_scores, target_far)

    if protocol == "tar_at_far":
        score_threshold = float(tar_far_result.threshold)
    elif protocol == "eer":
        score_threshold = float(thr_eer)
    else:
        raise ValueError(f"Unknown verification protocol '{protocol}'.")

    # Convert similarity-space decision threshold to distance space (1 - cosine)
    # for compatibility with the distance-based predict path.
    distance_threshold = float(1.0 - score_threshold)

    payload = {
        "dataset": dataset_name,
        "embedder": embedder_key,
        "distance_threshold": distance_threshold,
        "eer": float(eer),
        "eer_score_threshold": float(thr_eer),
        "far": float(far_at_thr),
        "frr": float(frr_at_thr),
        "roc_auc": float(roc_auc),
        "protocol": protocol,
        "score_threshold": score_threshold,
        "target_far": float(target_far),
        "tar_at_far_threshold": float(tar_far_result.threshold),
        "tar_at_far": float(tar_far_result.tar),
        "far_at_far": float(tar_far_result.far),
        "frr_at_far": float(tar_far_result.frr),
        "num_positive": int(pos_scores_arr.size),
        "num_negative": int(neg_scores_arr.size),
        "num_pairs_requested": num_pairs,
        "random_seed": random_seed,
        "plot_path": plot_file.name,
    }
    cache_store.save(payload)

    return VerificationThresholdResult(
        dataset=dataset_name,
        embedder=embedder_key,
        distance_threshold=distance_threshold,
        eer=float(eer),
        eer_score_threshold=float(thr_eer),
        far=float(far_at_thr),
        frr=float(frr_at_thr),
        roc_auc=float(roc_auc),
        cache_file=cache_file,
        plot_path=plot_file,
        protocol=protocol,
        score_threshold=score_threshold,
        target_far=float(target_far),
        tar_at_far_threshold=float(tar_far_result.threshold),
        tar_at_far=float(tar_far_result.tar),
        far_at_far=float(tar_far_result.far),
        frr_at_far=float(tar_far_result.frr),
    )
