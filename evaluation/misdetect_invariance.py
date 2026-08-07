"""Experiment 14 (pre-registered): 2D-input invariance of the frozen lifting
stage, through the real detection + lifting path.

Measures whether 2D keypoint corruption (displacement of distal or core
joints, confident scores left intact) creates a 3D corruption regime, and
whether the real detector confidence channel carries usable signal on clean
data.

Usage:
    python -m evaluation.misdetect_invariance
    python -m evaluation.misdetect_invariance --cameras 0,1 \
        --magnitudes 0.03,0.10,0.15 --selfcheck

Artifacts: thesis_artifacts/misdetect/misdetect.json
"""
import argparse
import json
import os
import sys
import time

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from backend.model_loader import get_model, get_detector
from evaluation.lifting import lift_from_coco_window
from evaluation.protocol import evaluated_centers, WINDOW, HALF

MPI_ROOT = "E:/Thesis/mpi_inf_3dhp"
SEQ = "S1/Seq1"

DISTAL = (7, 8, 9, 10, 13, 14, 15, 16)   # COCO: elbows, wrists, knees, ankles
CORE = (5, 6, 11, 12)                     # COCO: shoulders, hips
GROUPS = {"distal": DISTAL, "core": CORE}

CACHE = os.path.join(REPO_ROOT, "thesis_artifacts", "cross_view_eval",
                     "predictions_cache.npz")


def frame_dir(cam):
    return os.path.join(MPI_ROOT, SEQ, "frames_cam%d" % cam)


def detect_camera(detector, cam):
    import cv2
    paths = sorted(os.path.join(frame_dir(cam), f)
                   for f in os.listdir(frame_dir(cam)) if f.endswith(".jpg"))
    wh = None
    kpts = np.zeros((len(paths), 17, 2), dtype=np.float32)
    scores = np.zeros((len(paths), 17), dtype=np.float32)
    for i, p in enumerate(paths):
        frame = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        if wh is None:
            wh = (frame.shape[1], frame.shape[0])
        kpts[i], scores[i] = detector.detect_with_rotation(frame)
    return paths, kpts, scores, wh


def lift_all(model, kpts_seq, scores_seq, wh, centers):
    raws = np.zeros((len(centers), 17, 3), dtype=np.float64)
    for j, c in enumerate(centers):
        lo = c - HALF
        raws[j] = lift_from_coco_window(model, kpts_seq[lo:lo + WINDOW],
                                        scores_seq[lo:lo + WINDOW],
                                        wh[0], wh[1])
    return raws


def cluster_ci(x, n_boot=2000, seed=0):
    """Percentile bootstrap 95% CI over frames (single sequence: frames are
    the unit)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        means[b] = x[idx].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cameras", default="0,1")
    ap.add_argument("--magnitudes", default="0.03,0.10,0.15")
    ap.add_argument("--out", default=os.path.join(
        REPO_ROOT, "thesis_artifacts", "misdetect", "misdetect.json"))
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()

    cams = [int(c) for c in args.cameras.split(",")]
    mags = [float(m) for m in args.magnitudes.split(",")]

    if args.selfcheck:
        assert cams and all(0 <= c <= 8 for c in cams)
        assert mags and all(0.0 < m < 0.5 for m in mags)
        assert 0 < len(GROUPS)
        print("selfcheck OK")
        return

    t0 = time.time()
    print("loading models...", flush=True)
    model = get_model()
    detector = get_detector()

    results = {"cameras": cams, "magnitudes": mags,
               "groups": list(GROUPS), "per_camera": {}}
    sanity_max = 0.0

    for cam in cams:
        tag = "S1_Seq1_cam%d" % cam
        print("== camera %d ==" % cam, flush=True)
        paths, kpts, scores, wh = detect_camera(detector, cam)
        bbox_diag = float(np.sqrt(wh[0] ** 2 + wh[1] ** 2))
        centers = evaluated_centers(len(paths))
        print("  %d frames, %d centers, bbox_diag=%.0f px"
              % (len(paths), len(centers), bbox_diag), flush=True)

        t1 = time.time()
        raw_clean = lift_all(model, kpts, scores, wh, centers)
        print("  clean lift: %.1fs" % (time.time() - t1), flush=True)

        # sanity anchor vs cache
        cache = np.load(CACHE)
        if tag + "__raw" in cache.files:
            cached = cache[tag + "__raw"]
            n = min(len(centers), len(cached))
            anchor = float(np.linalg.norm(raw_clean[:n] - cached[:n], axis=2).mean())
            sanity_max = max(sanity_max, anchor)
            print("  anchor vs cache: %.2f mm" % anchor, flush=True)
            conf = cache[tag + "__components"][:, 0]
            rel = cache[tag + "__reliability"]
        else:
            conf, rel = None, None

        # detector channel on the fresh re-detection
        det_conf = scores.mean(axis=1)
        chan = {"det_conf_mean": float(det_conf.mean()),
                "det_conf_min": float(det_conf.min()),
                "frac_lt_0.9": float((det_conf < 0.9).mean())}
        if rel is not None:
            chan["cached_reliability_mean"] = float(rel.mean())
            chan["cached_reliability_min"] = float(rel.min())
            chan["cached_conf_mean"] = float(conf.mean())
        print("  detector channel: mean %.3f min %.3f frac<0.9 %.3f"
              % (chan["det_conf_mean"], chan["det_conf_min"],
                 chan["frac_lt_0.9"]), flush=True)

        rng = np.random.default_rng(42 + cam)
        dirs = rng.normal(size=(17, 2))
        dirs = dirs / np.linalg.norm(dirs, axis=1, keepdims=True)

        cam_res = {"n_frames": len(paths), "wh": list(wh),
                   "detector_channel": chan, "conditions": {}}
        for gname, group in GROUPS.items():
            for frac in mags:
                k = kpts.copy()
                k[:, group] += dirs[list(group)] * (frac * bbox_diag)
                np.clip(k, 0, (wh[0] - 1, wh[1] - 1), out=k)
                t2 = time.time()
                raw_c = lift_all(model, k, scores, wh, centers)
                per_frame = np.linalg.norm(raw_c - raw_clean, axis=2).mean(axis=1)
                lo, hi = cluster_ci(per_frame)
                mean_d = float(per_frame.mean())
                max_d = float(np.linalg.norm(raw_c - raw_clean, axis=2).max())
                g_d = float(np.linalg.norm(
                    raw_c[:, group] - raw_clean[:, group], axis=2).mean())
                cam_res["conditions"]["%s_f%.2f" % (gname, frac)] = {
                    "mean_mm": mean_d, "ci95_mm": [lo, hi],
                    "max_mm": max_d, "group_only_mm": g_d}
                print("  %-6s f=%.2f  mean %6.2f mm  [%5.2f, %5.2f]  max %6.2f  "
                      "group %6.2f  (%.1fs)"
                      % (gname, frac, mean_d, lo, hi, max_d, g_d,
                         time.time() - t2), flush=True)
        results["per_camera"][tag] = cam_res

    results["sanity_anchor_max_mm"] = sanity_max
    results["meta"] = {"elapsed_s": round(time.time() - t0, 1)}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print("\nWrote %s  (%.0fs total)" % (args.out, time.time() - t0), flush=True)

    # Reading verdict
    worst = max(c["conditions"]["%s_f%.2f" % (g, max(mags))]["mean_mm"]
                for cam_res in results["per_camera"].values()
                for g in GROUPS)
    sat = all(cam_res["detector_channel"]["det_conf_mean"] >= 0.99
              and cam_res["detector_channel"]["frac_lt_0.9"] <= 0.05
              for cam_res in results["per_camera"].values())
    p1 = worst < 3.0
    p2 = sat
    if p1 and p2:
        verdict = "Reading 1: 2D channel is inert; failure surface confined to the 3D alignment level"
    elif p1:
        verdict = "Reading 2: P1 holds, P2 fails (confidence varies on clean data)"
    else:
        verdict = "Reading 3: 2D displacement propagates >= 3.0 mm"
    print("\nVERDICT:", verdict, flush=True)
    results["verdict"] = verdict
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
