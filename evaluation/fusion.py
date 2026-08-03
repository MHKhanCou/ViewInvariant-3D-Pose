"""
Training-free multi-view selection and fusion via the canonical frame.

Canonicalization expresses every camera's prediction in the same body-fixed
frame, so predictions from N uncalibrated cameras become directly comparable
— and therefore selectable and fusable — with no training, no extrinsics,
and no correspondence search.

The reliability score (evaluation/reliability.py) supplies the weights and
the selection criterion, so the whole pipeline stays training-free.

Strategies (all take canonical poses + their reliabilities):
    single_mean        expected error of an arbitrary view (deployment baseline)
    reliability_pick   keep the single most reliable view
    median             coordinate-wise median across views (robust to outliers)
    weighted_mean      reliability-weighted average of all views
    topk_weighted      reliability-weighted average of the k most reliable views
    oracle_best        lowest-error view — REQUIRES ground truth, upper bound only
"""

import numpy as np

_EPS = 1e-8


def resolve_reflections(poses, weights=None):
    """
    Align mirror ambiguity across views before fusing.

    The body frame is built from a hip axis whose sign can flip when the
    subject's left/right ordering is ambiguous in a given view. Averaging a
    flipped pose with an unflipped one destroys the fusion, so each pose is
    compared against the reference (highest-weight view) and mirrored about
    the frame's x-axis if that reduces disagreement.

    Returns:
        aligned: (N, 17, 3) poses in a consistent frame
        n_flipped: int, how many views needed mirroring
    """
    poses = np.asarray(poses, dtype=np.float64)
    if len(poses) < 2:
        return poses, 0
    w = np.ones(len(poses)) if weights is None else np.asarray(weights)
    ref = poses[int(np.argmax(w))]

    aligned, n_flipped = [], 0
    for p in poses:
        mirrored = p.copy()
        mirrored[:, 0] *= -1
        if np.linalg.norm(mirrored - ref) < np.linalg.norm(p - ref):
            aligned.append(mirrored)
            n_flipped += 1
        else:
            aligned.append(p)
    return np.array(aligned), n_flipped


def reliability_pick(poses, reliabilities):
    """Most reliable single view. Training-free, no GT."""
    return np.asarray(poses)[int(np.argmax(reliabilities))]


def median_fuse(poses, reliabilities=None):
    """Coordinate-wise median — robust to a minority of bad views."""
    aligned, _ = resolve_reflections(poses, reliabilities)
    return np.median(aligned, axis=0)


def weighted_mean_fuse(poses, reliabilities):
    """Reliability-weighted mean of all views."""
    aligned, _ = resolve_reflections(poses, reliabilities)
    w = np.asarray(reliabilities, dtype=np.float64)
    w = w / max(w.sum(), _EPS)
    return (aligned * w[:, None, None]).sum(axis=0)


def topk_weighted_fuse(poses, reliabilities, k=3):
    """Reliability-weighted mean over the k most reliable views."""
    rel = np.asarray(reliabilities, dtype=np.float64)
    k = min(k, len(rel))
    sel = np.argsort(-rel)[:k]
    return weighted_mean_fuse(np.asarray(poses)[sel], rel[sel])


STRATEGIES = {
    "reliability_pick": lambda p, r: reliability_pick(p, r),
    "median": lambda p, r: median_fuse(p, r),
    "weighted_mean": lambda p, r: weighted_mean_fuse(p, r),
    "top3_weighted": lambda p, r: topk_weighted_fuse(p, r, 3),
    "top5_weighted": lambda p, r: topk_weighted_fuse(p, r, 5),
}


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    base = rng.standard_normal((17, 3))
    poses = np.array([base + rng.standard_normal((17, 3)) * s
                      for s in (0.01, 0.02, 0.30)])
    rels = np.array([0.95, 0.90, 0.40])

    assert np.allclose(reliability_pick(poses, rels), poses[0]), "pick failed"

    # The bad third view should not dominate a weighted fusion.
    err_w = np.linalg.norm(weighted_mean_fuse(poses, rels) - base)
    err_plain = np.linalg.norm(poses.mean(axis=0) - base)
    assert err_w < err_plain, "weighting did not help"

    # Reflection handling: a mirrored duplicate must fuse back to the original.
    mirrored = poses[0].copy()
    mirrored[:, 0] *= -1
    aligned, n = resolve_reflections(np.array([poses[0], mirrored]), [1.0, 1.0])
    assert n == 1, "mirror not detected"
    assert np.allclose(aligned[0], aligned[1]), "mirror not resolved"

    print("fusion.py self-checks passed")
