"""
The axis-length law at the joint level.

Pre-registered in thesis_artifacts/radial/PREREGISTRATION.md, committed before
this ran.

Section 5.16 tested d^2 ~= (c*r_bar/L)^2 + d0^2 BETWEEN frame constructions and
confirmed the mechanism. The conditioning experiment tested it BETWEEN FRAMES of
one construction and it failed, because there L is an anatomical near-constant.
This tests the level in between, which the derivation speaks to most directly and
which nothing in the report has touched: ACROSS JOINTS of one canonicalized pose.

A frame carrying rotation error Theta displaces a joint at p_j by Theta x p_j, so
the frame-induced part of the cross-view distance grows linearly in the joint's
radius. Procrustes removes exactly the rigid misalignment, so

    d_frame(j) ~= sqrt(d_canonical(j)^2 - d_oracle(j)^2)

and sigma, L_hip and L_torso are all measured elsewhere, making the predicted
slope band parameter-free.

Run:  ./venv/Scripts/python.exe -m evaluation.radial_law
      ./venv/Scripts/python.exe -m evaluation.radial_law --preds <cache> --tag <name>
"""

import argparse
import itertools
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from evaluation.h36m_crossview import EVAL_STRIDE, canonicalize_stream
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video, parse_video
from evaluation.oracle import procrustes_align

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "radial")
SEED = 12345
BOOTSTRAP_DRAWS = 10000

# Measured in other experiments; see the pre-registration. Nothing here is fitted
# to the data this module reads.
SIGMA_MM = 7.5        # limb-frame fit, Section 5.16
L_HIP_MM = 275.8      # 52900 frames, conditioning pre-registration
L_TORSO_MM = 456.1    # multiscale artifact
_ISO = np.sqrt(4.0 / 3.0)   # two independent views, isotropic axis orientation
SLOPE_LO = _ISO * (2 * SIGMA_MM / L_TORSO_MM)
SLOPE_HI = _ISO * np.sqrt((2 * SIGMA_MM / L_HIP_MM) ** 2
                          + (2 * SIGMA_MM / L_TORSO_MM) ** 2)

JOINT_NAMES = [
    "root", "r_hip", "r_knee", "r_foot", "l_hip", "l_knee", "l_foot",
    "spine", "thorax", "neck", "head", "l_shoulder", "l_elbow", "l_wrist",
    "r_shoulder", "r_elbow", "r_wrist",
]


def per_joint_oracle(pose_a, pose_b):
    """Per-joint distance after the optimal rigid rotation of a onto b.

    evaluation.oracle returns the mean over joints; the whole point here is the
    profile across joints, so the last step is redone rather than the helper
    edited.
    """
    R, _, _ = procrustes_align(pose_a, pose_b)
    aligned = (pose_a - pose_a.mean(axis=0)) @ R.T + pose_b.mean(axis=0)
    return np.linalg.norm(aligned - pose_b, axis=1)


def collect(videos, stride=EVAL_STRIDE):
    """Per-joint canonical and oracle distances, plus each joint's radius."""
    groups = {}
    for vid, v in videos.items():
        subj, action, cam = parse_video(vid)
        groups.setdefault((subj, action), {})[cam] = v

    n_pairs = 0
    sum_can = np.zeros(17)      # sum of squared per-joint canonical distance
    sum_orc = np.zeros(17)      # sum of squared per-joint oracle distance
    sum_r = np.zeros(17)        # sum of per-joint radius
    n_frames = 0
    # Bootstrap needs group-level sums, since the six pairs of a subject-action
    # group come from the same four camera streams of the same motion.
    per_group = []

    for (subj, action), cams in sorted(groups.items()):
        if len(cams) < 4:
            continue
        order = sorted(cams)
        n = min(len(cams[c]["pred"]) for c in order)
        sel = np.arange(0, n, stride)

        per_cam = {}
        for c in order:
            P = cams[c]["pred"][sel]
            P = P - P[:, 0:1, :]
            can, ok = canonicalize_stream(P)
            per_cam[c] = {"can": can, "ok": ok,
                          "r": np.linalg.norm(can, axis=2)}

        g_can = np.zeros(17)
        g_orc = np.zeros(17)
        g_r = np.zeros(17)
        g_n = 0
        for a, b in itertools.combinations(order, 2):
            A, B = per_cam[a], per_cam[b]
            keep = A["ok"] & B["ok"]
            for i in np.flatnonzero(keep):
                d_can = np.linalg.norm(A["can"][i] - B["can"][i], axis=1)
                d_orc = per_joint_oracle(A["can"][i], B["can"][i])
                g_can += d_can ** 2
                g_orc += d_orc ** 2
                # The radius the joint actually sits at, averaged over both views.
                g_r += 0.5 * (A["r"][i] + B["r"][i])
                g_n += 1
            n_pairs += 1

        if g_n:
            per_group.append({"can": g_can, "orc": g_orc, "r": g_r, "n": g_n})
            sum_can += g_can
            sum_orc += g_orc
            sum_r += g_r
            n_frames += g_n

    return {"sum_can": sum_can, "sum_orc": sum_orc, "sum_r": sum_r,
            "n": n_frames, "n_pairs": n_pairs, "groups": per_group}


def profile(sum_can, sum_orc, sum_r, n):
    """RMS per-joint distances and the frame-induced residual, per joint."""
    rms_can = np.sqrt(sum_can / n)
    rms_orc = np.sqrt(sum_orc / n)
    # The subtraction is a difference of squares, which is why the accumulators
    # are squared sums rather than sums of distances.
    frame = np.sqrt(np.maximum(0.0, sum_can - sum_orc) / n)
    return rms_can, rms_orc, frame, sum_r / n


def fit_through_origin(r, y):
    """Least squares slope with no intercept - the derivation has none.

    A joint at the root cannot be displaced by a rotation about the root, so
    d_frame(0) = 0 is a constraint of the theory rather than a convenience.
    """
    slope = float(np.sum(r * y) / np.sum(r * r))
    resid = y - slope * r
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return slope, r2


def analyse(data, draws=BOOTSTRAP_DRAWS):
    rms_can, rms_orc, frame, r_bar = profile(
        data["sum_can"], data["sum_orc"], data["sum_r"], data["n"])

    slope, r2 = fit_through_origin(r_bar, frame)
    slope_can, _ = fit_through_origin(r_bar, rms_can)
    slope_orc, _ = fit_through_origin(r_bar, rms_orc)

    # Cluster bootstrap over subject-action groups, same unit as everywhere else.
    rng = np.random.default_rng(SEED)
    groups = data["groups"]
    slopes = np.empty(draws)
    r2s = np.empty(draws)
    for t in range(draws):
        pick = rng.integers(0, len(groups), size=len(groups))
        c = np.zeros(17); o = np.zeros(17); rr = np.zeros(17); nn = 0
        for j in pick:
            g = groups[j]
            c += g["can"]; o += g["orc"]; rr += g["r"]; nn += g["n"]
        _, _, f_b, r_b = profile(c, o, rr, nn)
        slopes[t], r2s[t] = fit_through_origin(r_b, f_b)

    ci = [float(np.percentile(slopes, 2.5)), float(np.percentile(slopes, 97.5))]
    ratio = float(slope_can / slope_orc) if slope_orc > 0 else float("inf")

    return {
        "n_frame_pairs": int(data["n"]),
        "n_camera_pairs": int(data["n_pairs"]),
        "joints": [
            {"name": JOINT_NAMES[j], "radius_mm": float(r_bar[j]),
             "canonical_rms_mm": float(rms_can[j]),
             "oracle_rms_mm": float(rms_orc[j]),
             "frame_induced_mm": float(frame[j])}
            for j in range(17)
        ],
        "slope_frame_induced": slope,
        "slope_ci95": ci,
        "r2": r2,
        "predicted_band": [float(SLOPE_LO), float(SLOPE_HI)],
        "slope_canonical": slope_can,
        "slope_oracle": slope_orc,
        "control_ratio_canonical_over_oracle": ratio,
        "p1_linear_r2_at_least_0p80": bool(r2 >= 0.80),
        "p2_slope_in_predicted_band": bool(SLOPE_LO <= slope <= SLOPE_HI),
        "p3_oracle_slope_materially_smaller": bool(ratio >= 2.0),
        "post_hoc": post_hoc(rms_can, r_bar),
    }


# ---------------------------------------------------------------------------
# POST HOC. Not pre-registered. Written after seeing the table above, in the
# same manner as bone_consistency.py records that its signal was found by
# exploratory probing. It is reported as a mechanism for the failure, not as a
# confirmed result.
# ---------------------------------------------------------------------------

# Joints the Gram-Schmidt frame is built from, or that lie between them: the
# construction constrains these directly, so they cannot test a prediction about
# how the frame's rotation error propagates.
PINNED = [0, 1, 4, 7, 8]
# Free joints, split by whether the kinematic chain from the torso to them
# crosses a hinge.
TORSO_RIGID = [9, 10, 11, 14]                  # neck, head, shoulders
ARTICULATED = [2, 3, 5, 6, 12, 13, 15, 16]     # knees, feet, elbows, wrists
# Nearly equal radius, opposite side of a hinge - a natural matched comparison.
MATCHED = [(11, 5), (14, 2)]                   # shoulder vs knee, per side


def post_hoc(rms_can, r_bar):
    """Why the radius prediction fails: articulation, not distance from root."""
    matched = []
    for a, b in MATCHED:
        matched.append({
            "rigid": JOINT_NAMES[a], "articulated": JOINT_NAMES[b],
            "radius_rigid_mm": float(r_bar[a]),
            "radius_articulated_mm": float(r_bar[b]),
            "radius_gap_pct": float(abs(r_bar[a] - r_bar[b])
                                    / max(r_bar[a], r_bar[b]) * 100.0),
            "canonical_rigid_mm": float(rms_can[a]),
            "canonical_articulated_mm": float(rms_can[b]),
            "ratio": float(rms_can[b] / rms_can[a]) if rms_can[a] > 0 else float("inf"),
        })
    return {
        "note": "NOT pre-registered. Explains the failure of the radius model.",
        "mean_radius_torso_rigid_mm": float(r_bar[TORSO_RIGID].mean()),
        "mean_radius_articulated_mm": float(r_bar[ARTICULATED].mean()),
        "canonical_torso_rigid_mm": float(rms_can[TORSO_RIGID].mean()),
        "canonical_articulated_mm": float(rms_can[ARTICULATED].mean()),
        "matched_radius_pairs": matched,
        "pinned_joints_canonical_mm": {
            JOINT_NAMES[j]: float(rms_can[j]) for j in PINNED},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    path = args.preds or os.path.join(PRED_DIR, "preds.npz")
    pn = np.load(path)
    videos = aggregate_by_video(meta, pn, int(pn["n_clips"]))

    data = collect(videos)
    res = analyse(data)
    res["prediction_cache"] = os.path.basename(path)

    print("=" * 78)
    print("THE AXIS-LENGTH LAW ACROSS JOINTS")
    print("=" * 78)
    print("  %d frame-pairs over %d camera pairs\n"
          % (res["n_frame_pairs"], res["n_camera_pairs"]))
    print("  %-12s %9s %11s %9s %11s"
          % ("joint", "radius", "canonical", "oracle", "frame-part"))
    for j in res["joints"]:
        print("  %-12s %8.1f %10.1f %9.1f %10.1f"
              % (j["name"], j["radius_mm"], j["canonical_rms_mm"],
                 j["oracle_rms_mm"], j["frame_induced_mm"]))
    print("\n  frame-induced part vs radius, through the origin:")
    print("    slope        %.4f   95%% CI [%.4f, %.4f]"
          % (res["slope_frame_induced"], *res["slope_ci95"]))
    print("    predicted    [%.4f, %.4f]   (nothing fitted)"
          % tuple(res["predicted_band"]))
    print("    R2           %.3f" % res["r2"])
    print("\n  control - slope against radius:")
    print("    canonical    %.4f" % res["slope_canonical"])
    print("    oracle       %.4f   ratio %.2fx"
          % (res["slope_oracle"], res["control_ratio_canonical_over_oracle"]))
    print("\n  pre-registered predictions:")
    print("    1 linear in radius (R2 >= 0.80)   %s" % res["p1_linear_r2_at_least_0p80"])
    print("    2 slope in predicted band         %s" % res["p2_slope_in_predicted_band"])
    print("    3 control holds (>= 2x)           %s" % res["p3_oracle_slope_materially_smaller"])

    ph = res["post_hoc"]
    print("\n  POST HOC, not pre-registered - why radius is the wrong variable:")
    print("    torso-rigid joints   radius %.0f mm   canonical %.1f mm"
          % (ph["mean_radius_torso_rigid_mm"], ph["canonical_torso_rigid_mm"]))
    print("    articulated joints   radius %.0f mm   canonical %.1f mm"
          % (ph["mean_radius_articulated_mm"], ph["canonical_articulated_mm"]))
    for m in ph["matched_radius_pairs"]:
        print("    %-11s %5.1f mm  vs  %-9s %6.1f mm   %.2fx  (radii differ %.1f%%)"
              % (m["rigid"], m["canonical_rigid_mm"], m["articulated"],
                 m["canonical_articulated_mm"], m["ratio"], m["radius_gap_pct"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR,
                       "radial%s.json" % (("_" + args.tag) if args.tag else ""))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    print("\nSaved: %s" % out)


if __name__ == "__main__":
    main()
