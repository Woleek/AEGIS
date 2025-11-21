from collections import defaultdict
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


from typing import Dict, List, Optional, Sequence, Tuple

from ..models.base import load_insightface_detector


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denom == 0:
        return 1.0
    return 1.0 - float(np.dot(vec1, vec2) / denom)


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
                print(
                    f"Comparing query '{query_key}' to candidate '{candidate_key}' at rank {k_offset + 1}"
                )
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
    ) -> None:
        self.detection_model = load_insightface_detector(ctx_id=0)
        self.identity_lookup = identity_lookup
        self.real_images = real_images
        self.anon_images = anon_images
        self.real_paths = real_paths
        self.anon_paths = anon_paths

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

        # calculate utility metrics for each image pair
        emotion, gender, race, age = self._measure_utility()
        ssim = self._calculate_ssim()
        psnr = self._calculate_psnr()

        # Combine results into a DataFrame
        results = {
            "ssim": ssim,
            "psnr": psnr,
            "emotion_match": emotion,
            "gender_match": gender,
            "race_match": race,
            "age_diff": age,
        }
        return pd.DataFrame([results])

    def _calculate_ssim(self) -> float:
        # Calculate SSIM for each image pair
        scores: List[float] = []
        for key in tqdm(self.paired_keys, desc="Calculating SSIM"):
            real_img = self.real_images[key]
            anon_img = self.anon_images[key]

            real_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            anon_rgb = cv2.cvtColor(anon_img, cv2.COLOR_BGR2RGB)

            bboxes, _ = self.detection_model.detect(real_rgb, max_num=1)

            # crop both images to face region if detected
            if bboxes is not None and len(bboxes) > 0:
                x1, y1, x2, y2, _ = bboxes[0].astype(int)
                real_rgb = real_rgb[y1:y2, x1:x2]
                anon_rgb = anon_rgb[y1:y2, x1:x2]

            ssim_value = structural_similarity(
                real_rgb, anon_rgb, data_range=255.0, channel_axis=2
            )
            scores.append(float(ssim_value))

        if not scores:
            return float("nan")
        return float(np.mean(scores))

    def _calculate_psnr(self) -> float:
        # Calculate PSNR for each image pair
        scores: List[float] = []
        for key in tqdm(self.paired_keys, desc="Calculating PSNR"):
            real_img = self.real_images[key]
            anon_img = self.anon_images[key]

            real_rgb = cv2.cvtColor(real_img, cv2.COLOR_BGR2RGB)
            anon_rgb = cv2.cvtColor(anon_img, cv2.COLOR_BGR2RGB)

            bboxes, _ = self.detection_model.detect(real_rgb, max_num=1)

            # crop both images to face region if detected
            if bboxes is not None and len(bboxes) > 0:
                x1, y1, x2, y2, _ = bboxes[0].astype(int)
                real_rgb = real_rgb[y1:y2, x1:x2]
                anon_rgb = anon_rgb[y1:y2, x1:x2]

            psnr_value = peak_signal_noise_ratio(real_rgb, anon_rgb, data_range=255)
            scores.append(float(psnr_value))

        if not scores:
            return float("nan")
        return float(np.mean(scores))

    def _measure_utility(self) -> float:
        # Measure all utility metrics for each image pair
        emotion_matches: List[bool] = []
        gender_matches: List[bool] = []
        race_matches: List[bool] = []
        age_differences: List[float] = []

        # Get paths for paired images
        for real_path, anon_path in tqdm(
            self.paired_paths, desc="Measuring utility metrics"
        ):
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

            # Emotion match
            real_emotion = real_analysis[0]["dominant_emotion"]
            anon_emotion = anon_analysis[0]["dominant_emotion"]
            emotion_matches.append(real_emotion == anon_emotion)

            # Gender match
            real_gender = real_analysis[0]["dominant_gender"]
            anon_gender = anon_analysis[0]["dominant_gender"]
            gender_matches.append(real_gender == anon_gender)

            # Race match
            real_race = real_analysis[0]["dominant_race"]
            anon_race = anon_analysis[0]["dominant_race"]
            race_matches.append(real_race == anon_race)

            # Age difference
            real_age = real_analysis[0]["age"]
            anon_age = anon_analysis[0]["age"]
            age_differences.append(abs(real_age - anon_age))

        # Calculate average metrics
        emotion_match_rate = (
            float(np.mean(emotion_matches)) if emotion_matches else float("nan")
        )
        gender_match_rate = (
            float(np.mean(gender_matches)) if gender_matches else float("nan")
        )
        race_match_rate = float(np.mean(race_matches)) if race_matches else float("nan")
        age_difference = (
            float(np.mean(age_differences)) if age_differences else float("nan")
        )

        return emotion_match_rate, gender_match_rate, race_match_rate, age_difference


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
        self, threshold: Optional[float] = None
    ) -> Tuple[pd.DataFrame, float, float, float]:
        real_pairs, anon_pairs = self._build_pairs()
        if threshold is None:
            threshold = self._fit_threshold(real_pairs)
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

    def _fit_threshold(self, pairs: Sequence[Tuple]) -> float:
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

        scores = 1.0 - distances_arr

        # Call the new shared static method
        eer, threshold_score = self._compute_eer_from_scores(labels_arr, scores)

        # Handle the specific fallback case for this function
        if np.isinf(threshold_score):
            # Fallback: default to median distance if ROC could not produce finite thresholds.
            return float(np.median(distances_arr))

        # Store diagnostics for downstream reporting.
        self._last_fit_eer = float(eer)
        self._last_fit_eer_score_threshold = float(threshold_score)

        distance_threshold = float(1.0 - threshold_score)
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


def compute_or_load_verification_threshold(
    dataset_name: str,
    embedder_key: str,
    embeddings: Dict[str, np.ndarray],
    identity_lookup: DatasetIdentityLookup,
    thresholds_root: Path,
    num_pairs: int,
    random_seed: int,
    force_recompute: bool = False,
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
        if (
            cached
            and cached.get("dataset") == dataset_name
            and cached.get("embedder") == embedder_key
        ):
            plot_path = Path(cached.get("plot_path", plot_file))
            if not plot_path.is_absolute():
                plot_path = threshold_dir / plot_path
            return VerificationThresholdResult(
                dataset=cached.get("dataset", dataset_name),
                embedder=cached.get("embedder", embedder_key),
                distance_threshold=float(cached["distance_threshold"]),
                eer=float(cached["eer"]),
                eer_score_threshold=float(cached["eer_score_threshold"]),
                far=float(cached.get("far", 0.0)),
                frr=float(cached.get("frr", 0.0)),
                roc_auc=float(cached.get("roc_auc", 0.0)),
                cache_file=cache_file,
                plot_path=plot_path,
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

    thr_eer, eer, far_at_thr, frr_at_thr, roc_auc = plot_far_frr_with_eer(
        pos_scores_arr,
        neg_scores_arr,
        title=f"{dataset_name} verification FAR/FRR",
        save_path=plot_file,
    )

    distance_threshold = float(1.0 - thr_eer)

    payload = {
        "dataset": dataset_name,
        "embedder": embedder_key,
        "distance_threshold": distance_threshold,
        "eer": float(eer),
        "eer_score_threshold": float(thr_eer),
        "far": float(far_at_thr),
        "frr": float(frr_at_thr),
        "roc_auc": float(roc_auc),
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
    )
