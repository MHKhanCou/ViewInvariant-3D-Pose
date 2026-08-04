"""
Quantitative test of the axis-length law.

Sections 5.12 and 5.16 established an ORDERING: frames built from longer axes
are more consistent across views. That is an empirical observation. This tests
the stronger, quantitative statement the geometry actually predicts.

Derivation
----------
A direction estimated from two joints separated by L, each carrying independent
isotropic position noise of standard deviation sigma, has segment-vector noise
of variance 2*sigma^2 per component. Only the component perpendicular to the
segment rotates it, and that has two degrees of freedom, so E[|eps_perp|^2] =
4*sigma^2. For small angles the induced angular error is

    theta_rms  ~=  2*sigma / L                                            (1)

which is where the L^2 inverse-variance weighting used elsewhere comes from,
rather than being assumed.

A frame in error by theta displaces a joint at radius r from the root by about
r*theta. Two views canonicalized independently carry independent frame errors,
so their disagreement adds in quadrature. Writing d for the mean cross-view
joint distance and d0 for the part that does not come from the frame at all,
being the pose error the two views disagree on regardless,

    d^2  ~=  (c / L)^2  +  d0^2,        c = 2*sqrt(2)*sigma*r_bar          (2)

Equation (2) is falsifiable. It says d against 1/L is a straight line once
squared, with a positive intercept, and it predicts the value of d for an axis
length not used to fit it.

What is fitted
--------------
Nine (L, d) pairs from the segment-frame variants of Section 5.12, where the
same construction is applied to anatomically different segments with genuinely
different axis lengths, plus the global-frame variants of Section 5.16. Two free
parameters, c and d0, so seven residual degrees of freedom.

Run:  ./venv/Scripts/python.exe -m evaluation.axis_length_law
"""

import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.multiscale import SEGMENTS
from evaluation.h36m_multiscale import SEGMENTS_LONGAXIS
from evaluation.h36m_replication import OUT_DIR as PRED_DIR
from evaluation.h36m_replication import aggregate_by_video

OUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "axis_law")
ART = os.path.join(REPO_ROOT, "thesis_artifacts")


def mean_axis_length(poses, a, b):
    return float(np.mean(np.linalg.norm(poses[:, b] - poses[:, a], axis=-1)))


def mean_radius(poses, joint_ids, root):
    """
    Mean distance from a level's own root to the joints that level is scored on.

    This is r_bar in equation (2) and it is NOT a constant across levels, which
    a first version of this analysis wrongly assumed. A limb frame is scored on
    three joints close to its own root; the global frame is scored on seventeen
    joints spanning the whole body. Holding r_bar fixed compares levels whose
    lever arms differ by a factor of three and makes the law look false when it
    is only being misapplied.
    """
    d = np.linalg.norm(poses[:, joint_ids] - poses[:, root:root + 1], axis=-1)
    return float(np.mean(d))


def collect(pred_file, ms_file, ml_file):
    """Pair each frame definition with the length of the axis it is built from."""
    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    pn = np.load(os.path.join(PRED_DIR, pred_file) if not os.path.isabs(pred_file)
                 else pred_file)
    vids = aggregate_by_video(meta, pn, int(pn["n_clips"]))
    P = np.concatenate([v["pred"][::40] for v in vids.values()])

    with open(os.path.join(ART, ms_file)) as f:
        ms = json.load(f)["summary"]
    with open(os.path.join(ART, ml_file)) as f:
        ml = json.load(f)["summary"]

    pts = []
    # Segment frames, shipped and long-axis definitions. The primary axis is the
    # (y_from, y_to) pair, which is what the construction is most sensitive to.
    for table, tag in ((SEGMENTS, "shipped"), (SEGMENTS_LONGAXIS, "long-axis")):
        levels = (ms["per_level_distance_mm"] if tag == "shipped"
                  else ms["long_axis_variant"]["per_level_distance_mm"])
        for name, (ids, root, (y0, y1), _x) in table.items():
            pts.append({"label": "%s %s" % (name, tag),
                        "L": mean_axis_length(P, y0, y1),
                        "r": mean_radius(P, ids, root),
                        "d": levels[name]})

    # Global frame: the lateral axis is what varies between these two variants,
    # and it is scored on all seventeen joints about the pelvis.
    for variant, (a, b) in (("hip_only", (4, 1)), ("shoulder_only", (11, 14))):
        pts.append({"label": "global, %s" % variant,
                    "L": mean_axis_length(P, a, b),
                    "r": mean_radius(P, list(range(17)), 0),
                    "d": ml["variants"][variant]["mean_canonical_mm"]})
    for p in pts:
        p["r_over_L"] = p["r"] / p["L"]
    return pts


def fit(points):
    """
    Least squares for d^2 = (c/L)^2 + d0^2, which is linear in (c^2, d0^2).

    Fitting the squared form keeps the problem linear and avoids an iterative
    solver whose starting point would be a free choice.
    """
    x = np.array([p["r_over_L"] ** 2 for p in points])
    y = np.array([p["d"] ** 2 for p in points])
    A = np.vstack([x, np.ones_like(x)]).T
    (c2, d02), *_ = np.linalg.lstsq(A, y, rcond=None)
    c = float(np.sqrt(max(c2, 0.0)))
    d0 = float(np.sqrt(max(d02, 0.0)))
    pred = np.sqrt(np.maximum(
        (c * np.array([p["r_over_L"] for p in points])) ** 2 + d0 ** 2, 0))
    obs = np.array([p["d"] for p in points])
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    return {"c_mm2": c, "d0_mm": d0,
            "r2": float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
            "rmse_mm": float(np.sqrt(ss_res / len(obs))),
            "predicted": pred.tolist(), "observed": obs.tolist()}


def leave_one_out(points):
    """Refit without each point and predict it, which is the real test."""
    errs = []
    for i in range(len(points)):
        rest = [p for j, p in enumerate(points) if j != i]
        f = fit(rest)
        pred = np.sqrt((f["c_mm2"] * points[i]["r_over_L"]) ** 2 + f["d0_mm"] ** 2)
        errs.append({"label": points[i]["label"], "observed": points[i]["d"],
                     "predicted": float(pred),
                     "error_mm": float(pred - points[i]["d"])})
    return errs


def run(pred_file, ms_file, ml_file, name):
    pts = collect(pred_file, ms_file, ml_file)
    f = fit(pts)
    loo = leave_one_out(pts)
    mae = float(np.mean([abs(e["error_mm"]) for e in loo]))

    print("=" * 78)
    print("AXIS-LENGTH LAW  d^2 = (c/L)^2 + d0^2   [%s]" % name)
    print("=" * 78)
    print("  %d frame definitions, 2 free parameters\n" % len(pts))
    print("  %-26s %9s %9s %9s" % ("frame definition", "axis mm", "obs mm", "fit mm"))
    for p, pr in zip(pts, f["predicted"]):
        print("  %-26s %9.1f %9.1f %9.1f" % (p["label"], p["L"], p["d"], pr))
    print("\n  c  = %.0f mm^2      (= 2*sqrt(2)*sigma*r_bar)" % f["c_mm2"])
    print("  d0 = %.1f mm        irreducible, independent of the frame" % f["d0_mm"])
    print("  R^2 = %.3f   RMSE = %.2f mm" % (f["r2"], f["rmse_mm"]))
    print("  leave-one-out mean absolute error = %.2f mm" % mae)

    # What sigma does c imply? r_bar is the mean joint radius from the root.
    meta = np.load(os.path.join(PRED_DIR, "meta.npz"), allow_pickle=True)
    pn = np.load(os.path.join(PRED_DIR, pred_file) if not os.path.isabs(pred_file)
                 else pred_file)
    vids = aggregate_by_video(meta, pn, int(pn["n_clips"]))
    P = np.concatenate([v["pred"][::40] for v in vids.values()])
    r_bar = float(np.mean(np.linalg.norm(P - P[:, 0:1, :], axis=-1)))
    sigma = f["c_mm2"] / (2 * np.sqrt(2) * r_bar)
    print("\n  mean joint radius r_bar = %.1f mm" % r_bar)
    print("  implied per-joint noise sigma = %.1f mm" % sigma)
    print("  (a sanity check, not a fit: this should be the order of the "
          "backbone's own error)")

    out = {"points": pts, "fit": f, "leave_one_out": loo,
           "loo_mae_mm": mae, "r_bar_mm": r_bar, "implied_sigma_mm": sigma,
           "cache": pred_file}
    os.makedirs(OUT_DIR, exist_ok=True)
    p = os.path.join(OUT_DIR, "axis_law%s.json" % name)
    with open(p, "w") as fh:
        json.dump(out, fh, indent=2)
    print("\nSaved: %s" % p)
    return out


def main():
    run("preds.npz", "h36m_multiscale/h36m_multiscale.json",
        "multilandmark/results.json", "")
    print()
    run(os.path.join(ART, "h36m_motionbert", "preds_motionbert.npz"),
        "h36m_multiscale/h36m_multiscale_motionbert.json",
        "multilandmark/results_motionbert.json", "_motionbert")


if __name__ == "__main__":
    main()

