"""Face++ evaluation for commercial black-box transfer.

For each subject, compares a masked avatar render against a genuine reference
photo using the Face++ Compare API, recording the confidence score (0-100).

Reports, over all subjects:
  - Mean confidence  (lower = more private)
  - Verification rate (% with confidence >= Face++ threshold at the chosen FAR)

Requires FACEPP_API_KEY and FACEPP_API_SECRET (in .env).
"""
import argparse
import base64
import os
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

COMPARE_URL = "https://api-us.faceplusplus.com/facepp/v3/compare"
SAFE_SLEEP = 1.0       # free-tier throttle (~1 QPS)
MAX_RETRIES = 5
RETRY_SLEEP = 2.0
FAR_KEY = "1e-3"       # match the paper's TAR@FAR=1e-3 verification convention


def encode_image(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def post(url, payload):
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.post(url, data=payload, timeout=30).json()
        except Exception as e:  # network hiccup
            time.sleep(RETRY_SLEEP * (attempt + 1))
            continue
        msg = resp.get("error_message", "")
        if msg.startswith("CONCURRENCY_LIMIT_EXCEEDED") or msg.startswith("INTERNAL_ERROR"):
            time.sleep(RETRY_SLEEP * (attempt + 1))
            continue
        return resp
    return resp


def pick_reference(ref_dir: Path, subject: str, ref_frame: str | None) -> Path | None:
    """Return a single genuine reference image for the subject."""
    subj_dir = ref_dir / subject
    if subj_dir.is_dir():
        files = sorted(subj_dir.glob("*.png")) + sorted(subj_dir.glob("*.jpg"))
        if not files:
            return None
        if ref_frame is not None:
            match = [f for f in files if ref_frame in f.name]
            if match:
                return match[0]
        # default: first frame from the frontal camera "_08" if present, else first
        frontal = [f for f in files if "_08" in f.name]
        return frontal[0] if frontal else files[0]
    # flat layout fallback: <ref_dir>/<subject>.png
    for ext in ("png", "jpg", "jpeg"):
        p = ref_dir / f"{subject}.{ext}"
        if p.exists():
            return p
    return None


def compare(api_key, api_secret, img_a: Path, img_b: Path):
    resp = post(COMPARE_URL, {
        "api_key": api_key,
        "api_secret": api_secret,
        "image_base64_1": encode_image(img_a),
        "image_base64_2": encode_image(img_b),
    })
    if "error_message" in resp:
        return None, None, resp["error_message"]
    conf = resp.get("confidence")
    thr = (resp.get("thresholds") or {}).get(FAR_KEY)
    if conf is None:
        # a face was not detected in one of the images
        return None, thr, "NO_CONFIDENCE (face not detected?)"
    return float(conf), (float(thr) if thr is not None else None), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masked-dir", required=True, type=Path,
                    help="Flat dir of masked renders named <subject>.png")
    ap.add_argument("--reference-dir", required=True, type=Path,
                    help="Genuine reference dir (per-subject subdirs, or flat <subject>.png)")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--label", default="")
    ap.add_argument("--ref-frame", default=None,
                    help="Substring to select a specific reference frame filename")
    ap.add_argument("--subjects", nargs="*", default=None,
                    help="Explicit subject ids; default = all <subject>.png in masked-dir")
    args = ap.parse_args()

    api_key = os.environ["FACEPP_API_KEY"]
    api_secret = os.environ["FACEPP_API_SECRET"]

    if args.subjects:
        subjects = args.subjects
    else:
        subjects = sorted(p.stem for p in args.masked_dir.glob("*.png"))

    rows = []
    for subj in tqdm(subjects, desc=f"Face++ compare [{args.label}]"):
        masked = args.masked_dir / f"{subj}.png"
        ref = pick_reference(args.reference_dir, subj, args.ref_frame)
        if not masked.exists() or ref is None:
            rows.append({"subject": subj, "confidence": None, "threshold": None,
                         "verifies": None, "error": "missing image"})
            continue
        conf, thr, err = compare(api_key, api_secret, masked, ref)
        verifies = (conf is not None and thr is not None and conf >= thr)
        rows.append({"subject": subj, "reference": ref.name, "confidence": conf,
                     "threshold": thr, "verifies": verifies, "error": err})
        time.sleep(SAFE_SLEEP)

    df = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    valid = df[df["confidence"].notna()]
    n = len(valid)
    mean_conf = valid["confidence"].mean() if n else float("nan")
    ver_rate = 100.0 * valid["verifies"].mean() if n else float("nan")
    n_missing = len(df) - n
    print(f"\n=== {args.label} ===")
    print(f"subjects scored : {n}  (missing/undetected: {n_missing})")
    print(f"Mean confidence : {mean_conf:.2f}   (lower = more private)")
    print(f"Verification %  : {ver_rate:.2f}   (conf >= FAR={FAR_KEY} threshold; = 100 - PSR)")
    print(f"saved -> {args.output}")


if __name__ == "__main__":
    main()
