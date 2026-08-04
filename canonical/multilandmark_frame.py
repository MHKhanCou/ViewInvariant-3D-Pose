"""
Body frame estimated from redundant anatomical landmarks.

This is a test of the axis-length principle established in Section 5.12, not a
new algorithm. Least-squares body-frame fitting from shoulders, hips and ankles
is established practice in biomechanics; what is new here is only the question,
which is whether a principle discovered on limb frames also governs the global
frame.

Drop-in replacement for `canonical.body_frame.canonicalize_single`: same
signature, same return triple, same four metadata keys, so evaluation code can
switch between them by passing a different function.

Run:  ./venv/Scripts/python.exe -m canonical.multilandmark_frame
"""

import numpy as np

from .body_frame import canonicalize_single
from .multiscale import _gram_schmidt_frame

_EPS = 1e-8

# Vertical candidates: consecutive links up the spine plus the pelvis-to-thorax
# vector the baseline uses. All are torso-aligned and near-parallel, so their
# unit vectors can be averaged.
VERTICAL_LINKS = [(0, 7), (7, 8), (8, 9), (0, 8)]

# Lateral candidates. P[1]-P[4] is right-hip to left-hip and P[14]-P[11] is
# right-shoulder to left-shoulder, so both point the same anatomical way. This
# is why body_frame.py's existing fallback between them is sound, and it is what
# makes averaging them safe rather than cancelling.
LATERAL_PAIRS = [(1, 4), (14, 11)]

VARIANTS = ("both", "hip_only", "shoulder_only", "weighted", "svd")


def _weighted_direction(vectors):
    """
    Inverse-variance mean of unit vectors, with weights derived not tuned.

    If per-joint position noise is fixed and independent of segment length, the
    angular error of a direction estimated from a segment of length L scales as
    1/L, so its variance scales as 1/L^2 and inverse-variance weighting gives
    w = L^2. There is no free parameter here.
    """
    acc = np.zeros(3)
    for v in vectors:
        n = float(np.linalg.norm(v))
        if n < _EPS:
            continue
        acc += (v / n) * (n ** 2)
    return acc


def _spine_principal_direction(P):
    """
    Dominant direction of the spine chain by SVD.

    The five spine joints are close to collinear, so the first right-singular
    vector of the centred chain is a least-squares estimate of the body vertical
    that uses every joint rather than the two endpoints.
    """
    chain = P[[0, 7, 8, 9, 10]]
    centred = chain - chain.mean(axis=0)
    _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    d = Vt[0]
    # Sign is arbitrary from SVD; orient it head-upward using the chain itself.
    if np.dot(d, P[10] - P[0]) < 0:
        d = -d
    return d


def canonicalize_multilandmark(pose, variant="weighted", prev_z=None):
    """
    Canonicalize using axes estimated from several landmarks.

    Args:
        pose: (17, 3) root-relative pose, H36M-17 layout.
        variant: one of VARIANTS. "both" reproduces the two-vector baseline
            exactly and is delegated to `canonicalize_single` rather than
            reimplemented, so the comparison cannot drift.
        prev_z: (3,) previous frame's forward axis, for temporal sign
            consistency. Same meaning as in body_frame.canonicalize_single.

    Returns:
        (canonical_pose, R, metadata) with metadata keys "valid", "y_axis",
        "z_axis", "flipped", matching the baseline exactly.
    """
    if variant not in VARIANTS:
        raise ValueError("unknown variant %r; expected one of %s" % (variant, VARIANTS))
    if variant == "both":
        return canonicalize_single(pose, prev_z=prev_z)

    P = np.asarray(pose, dtype=np.float64)
    assert P.shape == (17, 3), "expected (17,3), got %s" % (P.shape,)
    P = P - P[0:1]

    if variant == "svd":
        y_raw = _spine_principal_direction(P)
    else:
        y_raw = _weighted_direction([P[b] - P[a] for a, b in VERTICAL_LINKS])

    if variant == "hip_only":
        x_raw = P[1] - P[4]
    elif variant == "shoulder_only":
        x_raw = P[14] - P[11]
    else:
        x_raw = _weighted_direction([P[a] - P[b] for a, b in LATERAL_PAIRS])

    R = _gram_schmidt_frame(y_raw, x_raw)
    fail = (np.zeros((17, 3), np.float32), np.eye(3, dtype=np.float32),
            {"valid": False, "y_axis": np.zeros(3, np.float32),
             "z_axis": np.zeros(3, np.float32), "flipped": False})
    if R is None:
        return fail

    x_body, y_body, z_body = R[:, 0], R[:, 1], R[:, 2]

    # Temporal sign consistency, identical to body_frame.py:114-120. Flipping
    # BOTH x and z preserves the determinant; flipping one would mirror the body.
    flipped = False
    if prev_z is not None and np.dot(z_body, prev_z) < 0:
        z_body, x_body = -z_body, -x_body
        flipped = True

    R = np.column_stack([x_body, y_body, z_body]).astype(np.float32)
    return (P.astype(np.float32) @ R, R,
            {"valid": True, "y_axis": y_body.astype(np.float32),
             "z_axis": z_body.astype(np.float32), "flipped": flipped})


if __name__ == "__main__":
    rng = np.random.default_rng(3)
    pose = np.zeros((17, 3))
    pose[1] = [-.13, -.02, 0]; pose[2] = [-.15, -.45, 0]; pose[3] = [-.16, -.90, 0]
    pose[4] = [.13, -.02, 0];  pose[5] = [.15, -.45, 0];  pose[6] = [.16, -.90, 0]
    pose[7] = [0, .25, 0];     pose[8] = [0, .50, 0]
    pose[9] = [0, .62, 0];     pose[10] = [0, .75, 0]
    pose[11] = [.18, .48, 0];  pose[12] = [.30, .25, 0];  pose[13] = [.35, .05, 0]
    pose[14] = [-.18, .48, 0]; pose[15] = [-.30, .25, 0]; pose[16] = [-.35, .05, 0]

    A = rng.standard_normal((3, 3))
    U, _, Vt = np.linalg.svd(A)
    Q = U @ Vt
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1

    for v in VARIANTS:
        a, _, ma = canonicalize_multilandmark(pose, v)
        b, _, mb = canonicalize_multilandmark(pose @ Q.T, v)
        assert ma["valid"] and mb["valid"], v
        d = float(np.abs(a - b).max())
        print("%-15s rotation invariance: %.2e" % (v, d))
        assert d < 1e-4, "%s is not rotation invariant" % v
    print("multilandmark_frame.py self-checks passed")
