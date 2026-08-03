"""
GT-anchored validation using MPI-INF-3DHP annot.mat (camera-coordinate annot3).

Answers the question the pure prediction-vs-prediction protocol cannot:
do our signals track ACTUAL prediction error?

1. Per-frame GT error: similarity-Procrustes (Umeyama: rotation + translation
   + scale) aligned distance between each cached prediction and its GT. The
   similarity alignment removes coordinate-convention and scale differences,
   per the examiner rule against convention-sensitive comparisons. This metric
   is NEVER called MPJPE.
2. Spearman rho(reliability, GT error) — does the training-free reliability
   score rank frames by actual error?
3. Spearman rho(cross-view distance, GT error) — does cross-view consistency
   (raw and canonical) track actual error?
4. GT bridge sanity: world-transformed GT from two cameras must agree (~0 mm).

Joint mapping: predictions use H36M-17 layout (see canonical/body_frame.py).
GT annot3 has 28 MPI joints. Direct map below; torso choices documented:
  H36M 7 (mid spine)   -> MPI 3  'spine'   (closest mid-torso point)
  H36M 8 (upper torso) -> MPI 1  'spine4'  (chest level)
  H36M 10 (head)       -> MPI 6  'head'
COCO-derived predictions synthesize torso joints by interpolation, so torso
correspondence is approximate either way; limbs and hips are exact.

Run:  ./venv/Scripts/python.exe -m evaluation.gt_eval
"""

import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from canonical.mpi_eval import load_annot_mat, load_camera_calibration, transform_to_world
from evaluation.protocol import MPI_ROOT, SEQUENCES, build_pairing_table
from evaluation.run_eval import load_cache, cam_key

OUTPUT_DIR = os.path.join(REPO_ROOT, "thesis_artifacts", "gt_validation")

# H36M-17 index -> MPI-28 index (see module docstring for torso choices).
H36M17_FROM_MPI28 = [4, 18, 19, 20, 23, 24, 25, 3, 1, 5, 6, 14, 15, 16, 9, 10, 11]


def similarity_align_error(source, target):
    """
    Mean per-joint distance after optimal similarity (Umeyama) alignment of
    source onto target: rotation + translation + isotropic scale.

    Args:
        source: (N, 3) prediction (arbitrary units).
        target: (N, 3) ground truth (mm).

    Returns:
        float, mean per-joint L2 in target units (mm).
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    mu_s, mu_t = source.mean(axis=0), target.mean(axis=0)
    Sc, Tc = source - mu_s, target - mu_t

    H = Sc.T @ Tc
    U, D, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    S_fix = np.diag([1.0, 1.0, d])
    R = Vt.T @ S_fix @ U.T

    var_s = (Sc ** 2).sum()
    scale = (D * np.diag(S_fix)).sum() / max(var_s, 1e-12)

    aligned = scale * Sc @ R.T + mu_t
    return float(np.mean(np.linalg.norm(aligned - target, axis=1)))


def load_gt17(subject, sequence):
    """
    Load GT annot3 for all cameras of a sequence, mapped to H36M-17 order.

    Returns:
        {camera_id: (n_frames, 17, 3) float64 mm, camera coordinates}
    """
    annot = load_annot_mat(os.path.join(MPI_ROOT, subject, sequence, "annot.mat"))
    out = {}
    n_cams = annot["annot3"].shape[0]
    for cam in range(n_cams):
        arr = annot["annot3"][cam][0]           # (n_frames, 84)
        joints28 = arr.reshape(len(arr), 28, 3)
        out[cam] = joints28[:, H36M17_FROM_MPI28, :].astype(np.float64)
    return out


def gt_bridge_sanity(subject, sequence, gt17, frame_idx=13):
    """World-transform GT from each camera pair and report mean disagreement (mm)."""
    calib = load_camera_calibration(
        os.path.join(MPI_ROOT, subject, sequence, "camera.calibration"))
    extr = {c["id"]: c["extrinsic"] for c in calib}
    cams = sorted(set(gt17.keys()) & set(extr.keys()))
    diffs = []
    for i in range(len(cams)):
        for j in range(i + 1, len(cams)):
            a, b = cams[i], cams[j]
            wa = transform_to_world(gt17[a][frame_idx], extr[a])
            wb = transform_to_world(gt17[b][frame_idx], extr[b])
            # transform_to_world applies rotation only; compare shapes via
            # root-centering to drop the unapplied translation component.
            wa = wa - wa[0:1]
            wb = wb - wb[0:1]
            diffs.append(float(np.mean(np.linalg.norm(wa - wb, axis=1))))
    return float(np.mean(diffs)) if diffs else None


def main():
    cache = load_cache()
    if not cache:
        print("ERROR: prediction cache missing — run evaluation.run_eval first.")
        return

    results = {"per_camera": {}, "per_pair": [], "pooled": {}, "sanity": {}}
    all_rel, all_err = [], []

    gt_by_seq = {}
    for subject, sequence in SEQUENCES:
        gt_by_seq[(subject, sequence)] = load_gt17(subject, sequence)
        sanity = gt_bridge_sanity(subject, sequence,
                                  gt_by_seq[(subject, sequence)])
        results["sanity"][f"{subject}/{sequence}"] = sanity
        print(f"GT bridge sanity {subject}/{sequence}: "
              f"cross-camera world GT disagreement (shape) = {sanity:.1f} mm")

    # --- Per-camera GT error + reliability correlation ---
    gt_errors = {}  # cam_key -> {center: err}
    for key, fields in cache.items():
        subject, sequence, cam_str = key.split("_", 2)
        cam = int(cam_str.replace("cam", ""))
        gt17 = gt_by_seq[(subject, sequence)][cam]

        errs = []
        for j, c in enumerate(fields["centers"]):
            errs.append(similarity_align_error(fields["raw"][j], gt17[c]))
        errs = np.array(errs)
        gt_errors[key] = dict(zip(fields["centers"].tolist(), errs.tolist()))

        rel = fields["reliability"]
        rho, pval = spearmanr(rel, errs)
        results["per_camera"][key] = {
            "n_frames": int(len(errs)),
            "gt_error_mean_mm": float(errs.mean()),
            "gt_error_std_mm": float(errs.std()),
            "spearman_reliability_vs_error": float(rho),
            "p_value": float(pval),
        }
        all_rel.extend(rel.tolist())
        all_err.extend(errs.tolist())
        print(f"{key}: GT err {errs.mean():.1f}±{errs.std():.1f} mm, "
              f"rho(rel, err) = {rho:+.3f} (p={pval:.3g})")

    rho_pooled, p_pooled = spearmanr(all_rel, all_err)
    results["pooled"]["spearman_reliability_vs_error"] = float(rho_pooled)
    results["pooled"]["p_value"] = float(p_pooled)
    results["pooled"]["n_frames"] = len(all_rel)
    print(f"\nPOOLED rho(reliability, GT error) = {rho_pooled:+.3f} "
          f"(p={p_pooled:.3g}, n={len(all_rel)})")

    # --- Per-pair: cross-view distance vs GT error ---
    mc_path = os.path.join(REPO_ROOT, "thesis_artifacts", "cross_view_eval",
                           "results_multicam.json")
    with open(mc_path) as f:
        pair_results = json.load(f)

    pooled_can, pooled_raw, pooled_pair_err = [], [], []
    for r in pair_results:
        ka = cam_key({"subject": r["subject"], "sequence": r["sequence"],
                      "camera": r["cam_a"]})
        kb = cam_key({"subject": r["subject"], "sequence": r["sequence"],
                      "camera": r["cam_b"]})
        centers = r["frame_centers"]
        pair_err = np.array([
            (gt_errors[ka][c] + gt_errors[kb][c]) / 2 for c in centers])
        can_d = np.array(r["per_frame_canonical"])
        raw_d = np.array(r["per_frame_raw"])

        rho_can, p_can = spearmanr(can_d, pair_err)
        rho_raw, p_raw = spearmanr(raw_d, pair_err)
        results["per_pair"].append({
            "pair": f"{r['subject']}/{r['sequence']} cam{r['cam_a']}-cam{r['cam_b']}",
            "is_dev_pair": r["is_dev_pair"],
            "spearman_canonical_dist_vs_gt_error": float(rho_can),
            "spearman_raw_dist_vs_gt_error": float(rho_raw),
            "p_canonical": float(p_can),
            "gt_error_mean_mm": float(pair_err.mean()),
        })
        pooled_can.extend(can_d.tolist())
        pooled_raw.extend(raw_d.tolist())
        pooled_pair_err.extend(pair_err.tolist())

    rho_c, p_c = spearmanr(pooled_can, pooled_pair_err)
    rho_r, p_r = spearmanr(pooled_raw, pooled_pair_err)
    results["pooled"]["spearman_canonical_dist_vs_gt_error"] = float(rho_c)
    results["pooled"]["spearman_raw_dist_vs_gt_error"] = float(rho_r)
    print(f"POOLED rho(canonical cross-view dist, GT error) = {rho_c:+.3f} (p={p_c:.3g})")
    print(f"POOLED rho(raw cross-view dist, GT error)       = {rho_r:+.3f} (p={p_r:.3g})")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, "gt_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
