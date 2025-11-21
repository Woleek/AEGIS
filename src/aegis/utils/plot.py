from pathlib import Path
import sys
from typing import Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import auc, roc_curve


def plot_single_frame(rgb):
    plt.imshow(np.clip(rgb.cpu().detach().numpy(), 0, 1))
    plt.axis("off")
    plt.show()


def plot_far_frr_with_eer(
    pos_scores: np.ndarray,
    neg_scores: np.ndarray,
    title: str,
    save_path: Path,
) -> Tuple[float, float, float, float, float]:
    """Plot FAR/FRR curves and return threshold diagnostics.

    Returns
    -------
    threshold_score : float
        Cosine similarity threshold at the equal error rate point.
    eer : float
        Equal error rate value.
    far_at_thr : float
        False accept rate when applying the threshold.
    frr_at_thr : float
        False reject rate when applying the threshold.
    roc_auc : float
        Area under the ROC curve for the sampled scores.
    """

    if pos_scores.size == 0 or neg_scores.size == 0:
        raise ValueError("Both positive and negative score arrays must be non-empty.")

    y_true = np.concatenate(
        [
            np.ones_like(pos_scores, dtype=int),
            np.zeros_like(neg_scores, dtype=int),
        ]
    )
    y_scores = np.concatenate([pos_scores, neg_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    frr = 1.0 - tpr
    diff = frr - fpr
    idx = np.where(np.sign(diff[1:]) != np.sign(diff[:-1]))[0]
    if idx.size == 0:
        j = int(np.argmin(np.abs(diff)))
        eer = (fpr[j] + frr[j]) / 2.0
        thr_eer = float(thresholds[j])
    else:
        j = int(idx[0])
        x0, y0, t0 = fpr[j], frr[j], thresholds[j]
        x1, y1, t1 = fpr[j + 1], frr[j + 1], thresholds[j + 1]
        denom = (y1 - y0) - (x1 - x0)
        if abs(denom) < 1e-12:
            t = 0.0
        else:
            t = (x0 - y0) / denom
        t = float(np.clip(t, 0.0, 1.0))
        eer = x0 + t * (x1 - x0)
        thr_eer = float(t0 + t * (t1 - t0))

    far_at_thr = float((neg_scores >= thr_eer).mean())
    frr_at_thr = float((pos_scores < thr_eer).mean())
    roc_auc = float(auc(fpr, tpr))

    try:
        import matplotlib.pyplot as plt  # Local import to avoid mandatory dependency at import time

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(thresholds, fpr, label="FAR")
        ax.plot(thresholds, frr, label="FRR")
        ax.axvline(thr_eer, linestyle="--", label=f"Threshold = {thr_eer:.3f}")
        ax.scatter([thr_eer], [eer], s=40, color="red", zorder=5)
        ax.text(
            thr_eer + 0.02,
            eer + 0.02,
            f"EER = {eer:.3f}",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
            fontsize=10,
        )
        ax.set_title(title)
        ax.set_xlabel("Threshold (cosine similarity)")
        ax.set_ylabel("Error rate")
        ax.set_ylim(-0.02, 1.02)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        fig.savefig(save_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
    except ImportError:
        print(
            "matplotlib is not installed; skipping FAR/FRR plot generation. "
            f"Expected to write plot to {save_path}",
            file=sys.stderr,
        )

    return thr_eer, float(eer), far_at_thr, frr_at_thr, roc_auc
