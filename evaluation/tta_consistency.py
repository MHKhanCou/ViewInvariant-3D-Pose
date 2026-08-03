"""
Test-time-augmentation dispersion as a training-free error predictor.

Question: does the disagreement a frozen monocular 3D pose model already
produces across cheap test-time augmentations predict its own error, from a
SINGLE camera?

The pipeline currently discards this disagreement twice over:
`detect_with_rotation` computes four rotated detections and keeps the argmax;
`lift_from_coco_window` computes flipped and unflipped predictions and returns
their mean. Both are harvested here at essentially zero extra cost.

Criterion and predictors are fixed in advance — see
`thesis_artifacts/tta/PREREGISTRATION.md`, committed before this file existed.

NOT claimed: that canonicalization enables the comparison. Verified in source
that every arm already returns in the same camera frame (`_unrotate_kpts` maps
rotated detections back; the flip branch is un-mirrored by `flip_data` before
averaging). Canonicalization is used only for the per-joint dispersion vector
and the anatomical decomposition, both flagged exploratory.

Run:  ./venv/Scripts/python.exe -m evaluation.tta_consistency [--limit N]
"""

import argparse
import json
import os
import sys

import cv2
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.body_frame import canonicalize_single
from evaluation import corruptions
from evaluation.fusion import resolve_reflections
from evaluation.gt_eval import load_gt17, similarity_align_error
from evaluation.lifting import flip_data, lift_from_coco_window
from evaluation.protocol import HALF, WINDOW, discover_cameras, evaluated_centers
from evaluation.run_eval import load_cache as load_pred_cache

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "tta")
CACHE_PATH = os.path.join(OUT_DIR, "tta_cache.npz")
SEED = 12345

# Primary arm set: K = 6, constant on every frame, zero extra YOLO passes.
JITTER_SIGMAS = [0.0, 0.005, 0.01]     # fraction of bbox diagonal
FLIP_ARMS = ["nonflip", "flip"]
K_PRIMARY = len(JITTER_SIGMAS) * len(FLIP_ARMS)

PASS_RHO_FLOOR = 0.30                  # pre-registered usefulness floor
BOOTSTRAP_DRAWS = 10000


def cam_key(cam):
    cond = cam.get("condition", "static")
    suffix = "" if cond == "static" else f"_{cond}"
    return (f"{cam['subject']}_{cam['sequence'].replace('/', '_')}"
            f"_cam{cam['camera']}{suffix}")


def parse_key(key):
    """
    Split a cache key into (subject, sequence, camera, condition).

    Uses the prefix/suffix idiom from fusion_eval.collect, NOT
    gt_eval's split("_", 2) which mis-parses '..._cam0_dynamic'.
    """
    parts = key.split("_")
    subject, sequence = parts[0], parts[1]
    cam = int(parts[2].replace("cam", ""))
    condition = parts[3] if len(parts) > 3 else "static"
    return subject, sequence, cam, condition


# --------------------------------------------------------------------------
# dispersion metrics
# --------------------------------------------------------------------------

def disp_procrustes(arms):
    """PRIMARY. Mean pairwise similarity-aligned distance across arms (mm-like).

    Uses the same metric as the GT error, so the two correlate on equal terms.
    """
    n = len(arms)
    if n < 2:
        return np.nan
    d = [similarity_align_error(arms[i], arms[j])
         for i in range(n) for j in range(i + 1, n)]
    return float(np.mean(d))


def disp_axis_split(arms):
    """
    EXPLORATORY mechanism test, in CAMERA coordinates (not body coordinates):
    dispersion along camera depth (z) vs in-plane (x, y).

    The hypothesis is about depth ambiguity; canonicalize_single's z is the
    body sagittal axis, so this decomposition deliberately avoids it.
    """
    A = np.asarray(arms, dtype=np.float64)
    if len(A) < 2:
        return np.nan, np.nan
    spread = A.std(axis=0)                      # (17, 3) per-joint per-axis
    depth = float(np.mean(spread[:, 2]))
    inplane = float(np.mean(np.linalg.norm(spread[:, :2], axis=1)))
    return depth, inplane


def disp_scale(arms):
    """EXPLORATORY. Coefficient of variation of each arm's RMS joint norm."""
    norms = [float(np.sqrt(np.mean(np.sum(np.asarray(a) ** 2, axis=1))))
             for a in arms]
    m = float(np.mean(norms))
    return float(np.std(norms) / m) if m > 1e-9 else np.nan


def disp_per_joint_canonical(arms):
    """
    EXPLORATORY. Per-joint dispersion in the canonical body frame — the one
    place canonicalization genuinely helps, by making the (17,) vector
    comparable across cameras and subjects.

    Returns (17,) or None if fewer than two arms canonicalize validly.
    """
    canon = []
    for a in arms:
        c, _, meta = canonicalize_single(np.asarray(a, dtype=np.float32))
        if meta["valid"]:
            canon.append(c.astype(np.float64))
    if len(canon) < 2:
        return None
    # resolve_reflections negates x WITHOUT the left/right index swap that
    # lifting.flip_data performs. That is correct for a body-frame sign flip
    # but does NOT undo an anatomical mirror; mirror_would_help below counts
    # how often the flip_data convention would have been closer.
    aligned, _ = resolve_reflections(np.array(canon))
    return aligned.std(axis=0).mean(axis=1)


def mirror_diagnostic(arms):
    """Count frames where anatomical mirroring beats a plain sign flip."""
    canon = []
    for a in arms:
        c, _, meta = canonicalize_single(np.asarray(a, dtype=np.float32))
        if meta["valid"]:
            canon.append(c.astype(np.float64))
    if len(canon) < 2:
        return False
    ref = canon[0]
    hits = 0
    for c in canon[1:]:
        sign = c.copy()
        sign[:, 0] *= -1
        anat = flip_data(c.copy())
        if np.linalg.norm(anat - ref) < np.linalg.norm(sign - ref):
            hits += 1
    return hits > 0


# --------------------------------------------------------------------------
# per-camera pass
# --------------------------------------------------------------------------

def process_camera(model, detector, cam, limit=None, use_rotation=False):
    """Build TTA arms per evaluated centre and compute dispersion metrics."""
    paths = cam["frame_paths"]
    n = len(paths)
    centers = evaluated_centers(n)
    if limit:
        centers = centers[:limit]
    if not centers:
        return None

    need = range(min(centers) - HALF, max(centers) + HALF + 1)
    kpts0 = np.zeros((n, 17, 2), dtype=np.float32)
    scores0 = np.zeros((n, 17), dtype=np.float32)
    kpts_rot = np.zeros((4, n, 17, 2), dtype=np.float32)
    detected = np.zeros((4, n), dtype=bool)
    angles = [0, 90, 180, 270]
    wh = None

    for i in need:
        frame = cv2.cvtColor(cv2.imread(paths[i]), cv2.COLOR_BGR2RGB)
        if wh is None:
            wh = (frame.shape[1], frame.shape[0])
        k, s, per_angle = detector.detect_with_rotation(frame, return_all=True)
        if k is not None:
            kpts0[i], scores0[i] = k, s
        for ai, a in enumerate(angles):
            pk, _ = per_angle[a]
            if pk is not None:
                kpts_rot[ai, i] = pk
                detected[ai, i] = True
    W, H = wh

    rows = []
    n_skipped = 0
    rot_frames_ok = 0
    for c in centers:
        lo = c - HALF
        arms, arm_labels = [], []

        # --- primary arms: 3 jitter levels x 2 flip branches, K = 6 ---
        for si, sigma in enumerate(JITTER_SIGMAS):
            win_k = kpts0[lo:lo + WINDOW].copy()
            win_s = scores0[lo:lo + WINDOW].copy()
            if sigma > 0:
                rng = np.random.default_rng(SEED + 1000 * c + si)
                for t in range(WINDOW):
                    win_k[t], win_s[t] = corruptions.jitter(
                        win_k[t], win_s[t], sigma, rng)
            _, flip_arms = lift_from_coco_window(
                model, win_k, win_s, W, H, return_arms=True)
            for name in FLIP_ARMS:
                arms.append(flip_arms[name])
                arm_labels.append(f"jit{si}_{name}")

        if len(arms) < K_PRIMARY:
            n_skipped += 1
            continue

        # --- secondary rotation arms, only if every angle detected in-window ---
        rot_arms = []
        if use_rotation:
            ok = all(detected[ai, lo:lo + WINDOW].all() for ai in range(4))
            if ok:
                rot_frames_ok += 1
                for ai in range(1, 4):   # angle 0 already covered
                    _, fa = lift_from_coco_window(
                        model, kpts_rot[ai, lo:lo + WINDOW],
                        scores0[lo:lo + WINDOW], W, H, return_arms=True)
                    rot_arms.extend(fa[name] for name in FLIP_ARMS)

        depth, inplane = disp_axis_split(arms)
        pj = disp_per_joint_canonical(arms)
        rows.append({
            "center": int(c),
            "disp_procrustes": disp_procrustes(arms),
            "disp_depth": depth,
            "disp_inplane": inplane,
            "disp_scale": disp_scale(arms),
            "disp_per_joint_mean": float(np.mean(pj)) if pj is not None else np.nan,
            "disp_rotation": disp_procrustes(arms + rot_arms) if rot_arms else np.nan,
            "mirror_would_help": bool(mirror_diagnostic(arms)),
            "arm0_raw": arms[0],
            "n_arms": len(arms),
        })

    return {"rows": rows, "n_skipped": n_skipped,
            "rot_frames_ok": rot_frames_ok, "n_centers": len(centers)}


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------

def partial_spearman(x, y, z):
    """Spearman rho(x, y | z) via rank residuals."""
    from scipy.stats import rankdata
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a, b):
        b1 = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(b1, a, rcond=None)
        return a - b1 @ coef

    return float(spearmanr(resid(rx, rz), resid(ry, rz))[0])


def cluster_bootstrap_delta(per_cam, draws=BOOTSTRAP_DRAWS, seed=SEED):
    """
    95% CI on |rho(disp)| - |rho(reliability)|, resampling CAMERAS.

    Frames within a camera share overlapping 27-frame windows, so an i.i.d.
    frame bootstrap would be anticonservative; the camera is the unit.
    """
    rng = np.random.default_rng(seed)
    cams = list(per_cam)
    deltas = []
    for _ in range(draws):
        pick = rng.choice(len(cams), size=len(cams), replace=True)
        d, r, e = [], [], []
        for i in pick:
            c = per_cam[cams[i]]
            d.extend(c["disp"]); r.extend(c["rel"]); e.extend(c["err"])
        if len(set(e)) < 3:
            continue
        rd = spearmanr(d, e)[0]
        rr = spearmanr(r, e)[0]
        if not (np.isnan(rd) or np.isnan(rr)):
            deltas.append(abs(rd) - abs(rr))
    if not deltas:
        return None, None
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))


def main():
    ap = argparse.ArgumentParser(description="TTA dispersion experiment")
    ap.add_argument("--limit", type=int, default=None,
                    help="max centres per camera (smoke test)")
    ap.add_argument("--rotation", action="store_true",
                    help="include the secondary rotation arms")
    args = ap.parse_args()

    from backend.model_loader import get_detector, get_model
    model, detector = get_model(), get_detector()
    pred_cache = load_pred_cache()

    gts, per_cam, strata = {}, {}, {}
    diag = {"mirror_frames": 0, "total_frames": 0, "skipped": 0,
            "rotation_frames_ok": 0}

    for cam in discover_cameras():
        key = cam_key(cam)
        subject, sequence, cnum, condition = parse_key(key)
        print(f"  {key} ...", flush=True)
        res = process_camera(model, detector, cam, args.limit, args.rotation)
        if not res or not res["rows"]:
            continue
        if (subject, sequence) not in gts:
            gts[(subject, sequence)] = load_gt17(subject, sequence)
        # load_gt17 returns {camera_id: (n_frames, 17, 3)} — index camera first.
        gt_cam = gts[(subject, sequence)][cnum]

        disp, err, rel, conf = [], [], [], []
        cached = pred_cache.get(key)
        for row in res["rows"]:
            c = row["center"]
            disp.append(row["disp_procrustes"])
            err.append(similarity_align_error(row["arm0_raw"], gt_cam[c]))
            if cached is not None:
                idx = np.where(cached["centers"] == c)[0]
                if len(idx):
                    rel.append(float(cached["reliability"][idx[0]]))
                    conf.append(float(cached["components"][idx[0], 4]))
                    continue
            rel.append(np.nan)
            conf.append(np.nan)
            diag["mirror_frames"] += int(row["mirror_would_help"])
        diag["total_frames"] += len(res["rows"])
        diag["skipped"] += res["n_skipped"]
        diag["rotation_frames_ok"] += res["rot_frames_ok"]

        per_cam[key] = {"disp": disp, "err": err, "rel": rel, "conf": conf,
                        "condition": condition, "subject": subject,
                        "rows": res["rows"]}
        stratum = f"{subject}_{condition}"
        strata.setdefault(stratum, {"disp": [], "err": []})
        strata[stratum]["disp"].extend(disp)
        strata[stratum]["err"].extend(err)

    if not per_cam:
        print("No cameras processed.")
        return

    pooled_d = [v for c in per_cam.values() for v in c["disp"]]
    pooled_e = [v for c in per_cam.values() for v in c["err"]]
    pooled_r = [v for c in per_cam.values() for v in c["rel"]]
    pooled_c = [v for c in per_cam.values() for v in c["conf"]]

    rho_disp = float(spearmanr(pooled_d, pooled_e)[0])
    mask = ~np.isnan(pooled_r)
    rho_rel = float(spearmanr(np.array(pooled_r)[mask],
                              np.array(pooled_e)[mask])[0]) if mask.any() else np.nan
    partial = (partial_spearman(np.array(pooled_d)[mask],
                                np.array(pooled_e)[mask],
                                np.array(pooled_c)[mask])
               if mask.any() and not np.isnan(np.array(pooled_c)[mask]).all() else np.nan)

    scored = {k: v for k, v in per_cam.items() if not np.isnan(v["rel"]).all()}
    lo, hi = cluster_bootstrap_delta(scored) if scored else (None, None)

    strat_rho = {s: float(spearmanr(v["disp"], v["err"])[0])
                 for s, v in strata.items() if len(v["disp"]) > 5}

    crit_a = rho_disp >= PASS_RHO_FLOOR
    signs = {np.sign(r) for r in strat_rho.values() if not np.isnan(r)}
    crit_b = len(signs) == 1
    crit_c = (lo is not None and (lo > 0 or hi < 0))
    passed = bool(crit_a and crit_b and crit_c)

    print("\n" + "=" * 72)
    print("TTA DISPERSION vs GT ERROR")
    print("=" * 72)
    print(f"pooled rho(disp_procrustes, error) = {rho_disp:+.3f}  (n={len(pooled_d)})")
    print(f"pooled rho(reliability, error)     = {rho_rel:+.3f}   [incumbent]")
    print(f"partial rho(disp, err | detector_confidence) = {partial:+.3f}")
    for s, r in sorted(strat_rho.items()):
        print(f"  stratum {s:16s} rho = {r:+.3f}")
    if lo is not None:
        print(f"cluster bootstrap 95% CI on |rho_disp|-|rho_rel|: [{lo:+.3f}, {hi:+.3f}]")
    print(f"\n(a) rho >= {PASS_RHO_FLOOR}:            {crit_a}")
    print(f"(b) consistent sign across strata: {crit_b}  {sorted(strat_rho)}")
    print(f"(c) CI excludes 0:                 {crit_c}")
    print(f"\nPRE-REGISTERED VERDICT: {'PASS' if passed else 'FAIL'}"
          f"{'' if passed else '  -> report as falsification axis 5'}")

    out = {
        "preregistration": "thesis_artifacts/tta/PREREGISTRATION.md",
        "verdict_pass": passed,
        "criteria": {"rho_floor_met": bool(crit_a),
                     "sign_consistent": bool(crit_b),
                     "ci_excludes_zero": bool(crit_c)},
        "pooled": {
            "n": len(pooled_d),
            "spearman_disp_vs_error": rho_disp,
            "spearman_reliability_vs_error": rho_rel,
            "partial_spearman_given_detector_confidence": partial,
            "bootstrap_ci_delta_abs_rho": [lo, hi],
        },
        "per_stratum": strat_rho,
        "per_camera": {k: {"n": len(v["disp"]),
                           "spearman_disp_vs_error": float(spearmanr(v["disp"], v["err"])[0])}
                       for k, v in per_cam.items()},
        "diagnostics": diag,
        "exploratory": True,
        "arms": {"K_primary": K_PRIMARY, "jitter_sigmas": JITTER_SIGMAS,
                 "flip_arms": FLIP_ARMS, "rotation_included": bool(args.rotation)},
    }
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "tta_results.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
