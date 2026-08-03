"""
Controlled degradation operators for reliability calibration.

Each operator perturbs COCO-17 2D detections *before* the COCO->H36M
conversion, so the degradation propagates through exactly the same code path
that real detector noise does. Corrupting at the COCO level (rather than on
the 3D output or the H36M skeleton) matters for two reasons:

1. The four virtual H36M joints (pelvis, spine, thorax, head-top) are derived
   from COCO joints, so a corrupted shoulder correctly propagates into a
   corrupted thorax -- as it would with a genuinely bad detection.
2. `reliability.compute_reliability_score` indexes `scores_2d` in COCO-17
   order. Corrupting here keeps that contract intact.

Every operator maps to a documented real-world failure mode; none is an
arbitrary mathematical perturbation:

    jitter              -> detector localization noise, motion blur
    joint_dropout       -> occlusion, truncation at the frame edge
    limb_foreshorten    -> oblique / top-down camera placement
    hip_axis_collapse   -> frontal-lying poses, legs together
    torso_axis_collapse -> overhead camera, deep crouch

The last two directly attack the two vectors the Gram-Schmidt body frame is
built from, and are the operators expected to trigger the hard geometric
gates. Severity 0 is always the identity, so every sweep contains its own
uncorrupted control.

All operators are deterministic given a seeded RNG.
"""

import numpy as np

# COCO-17 keypoint indices.
NOSE = 0
L_EYE, R_EYE, L_EAR, R_EAR = 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Dropout order: most structurally load-bearing joints first. Removing hips
# and shoulders attacks the body frame; removing wrists/ears barely touches it.
# This ordering makes low-k dropout a *targeted* stressor rather than a
# lottery, so the severity axis stays monotonic.
DROPOUT_ORDER = [
    L_HIP, R_HIP,           # hip axis
    L_SHOULDER, R_SHOULDER,  # thorax (and the shoulder-axis fallback)
    L_KNEE, R_KNEE,
    NOSE,
    L_ANKLE, R_ANKLE,
    L_ELBOW, R_ELBOW,
    L_WRIST, R_WRIST,
    L_EYE, R_EYE, L_EAR, R_EAR,
]

LOWER_BODY = [L_KNEE, R_KNEE, L_ANKLE, R_ANKLE]


def _bbox_diagonal(kpts, scores, min_score=0.1):
    """
    Diagonal of the tight bbox over confident keypoints.

    Used as the length unit for jitter so that a given sigma means the same
    thing on a person filling the frame and a person 40 pixels tall. Falls
    back to the full keypoint spread, then to 1.0, when too few joints are
    confident.
    """
    valid = scores >= min_score
    pts = kpts[valid] if valid.sum() >= 2 else kpts
    if len(pts) < 2:
        return 1.0
    spread = pts.max(axis=0) - pts.min(axis=0)
    diag = float(np.linalg.norm(spread))
    return diag if diag > 1e-6 else 1.0


def jitter(kpts, scores, severity, rng):
    """
    Isotropic Gaussian noise on keypoint locations.

    `severity` is the standard deviation as a fraction of the person's bbox
    diagonal, making it scale-invariant across frames. Confidence scores are
    left untouched: a detector that is *wrong* is generally not aware it is
    wrong, which is precisely the failure mode Khanal & Zhou (2026) document
    for learned confidence. Leaving scores clean here means the detector-
    confidence component of the reliability score gets no free signal, so any
    correlation we measure must come from the geometric components.
    """
    if severity <= 0:
        return kpts.copy(), scores.copy()
    sigma = severity * _bbox_diagonal(kpts, scores)
    noise = rng.normal(0.0, sigma, size=kpts.shape)
    return (kpts + noise).astype(np.float32), scores.copy()


def joint_dropout(kpts, scores, severity, rng):
    """
    Zero out the `severity` most structurally important joints.

    Both the coordinates and the confidences are zeroed, which is how the
    upstream detector reports a missing joint. Deterministic (structure-first
    ordering) rather than random, so severity is monotonic.
    """
    k = int(severity)
    if k <= 0:
        return kpts.copy(), scores.copy()
    kpts_out = kpts.copy()
    scores_out = scores.copy()
    for idx in DROPOUT_ORDER[:k]:
        kpts_out[idx] = 0.0
        scores_out[idx] = 0.0
    return kpts_out, scores_out


def limb_foreshorten(kpts, scores, severity, rng):
    """
    Contract the lower limbs toward the hip midpoint by factor `severity`.

    Simulates an oblique or top-down camera, where legs project to a fraction
    of their true length. severity=1.0 is the identity; severity=0.1 collapses
    the legs almost entirely into the hips.
    """
    f = float(severity)
    if f >= 1.0:
        return kpts.copy(), scores.copy()
    hip_mid = 0.5 * (kpts[L_HIP] + kpts[R_HIP])
    kpts_out = kpts.copy()
    for idx in LOWER_BODY:
        kpts_out[idx] = hip_mid + f * (kpts[idx] - hip_mid)
    return kpts_out, scores.copy()


def hip_axis_collapse(kpts, scores, severity, rng):
    """
    Contract the left/right hips toward their midpoint by factor `severity`.

    Directly attacks `x_raw = P[1] - P[4]`, the horizontal vector of the
    Gram-Schmidt body frame. Physically this is a person viewed frontally with
    legs together, or a near-degenerate hip detection. Expected to trigger the
    `hip_axis_too_short` gate at high severity.
    """
    f = float(severity)
    if f >= 1.0:
        return kpts.copy(), scores.copy()
    hip_mid = 0.5 * (kpts[L_HIP] + kpts[R_HIP])
    kpts_out = kpts.copy()
    kpts_out[L_HIP] = hip_mid + f * (kpts[L_HIP] - hip_mid)
    kpts_out[R_HIP] = hip_mid + f * (kpts[R_HIP] - hip_mid)
    return kpts_out, scores.copy()


def torso_axis_collapse(kpts, scores, severity, rng):
    """
    Contract the shoulder girdle toward the hip midpoint by factor `severity`.

    Attacks `y_raw = P[8] - P[0]` (thorax minus root); the H36M thorax is
    derived from the COCO shoulders, so pulling the shoulders down onto the
    hips shortens the body vertical axis. Physically: an overhead camera, or a
    deep crouch. Expected to trigger `torso_axis_too_short`, and -- because a
    short torso axis also destabilizes the cross product -- possibly
    `axes_nearly_collinear`.
    """
    f = float(severity)
    if f >= 1.0:
        return kpts.copy(), scores.copy()
    hip_mid = 0.5 * (kpts[L_HIP] + kpts[R_HIP])
    kpts_out = kpts.copy()
    for idx in (L_SHOULDER, R_SHOULDER):
        kpts_out[idx] = hip_mid + f * (kpts[idx] - hip_mid)
    return kpts_out, scores.copy()


# Registry: name -> (operator, severity levels, units).
# Severity level 0 is the identity for every operator, giving each sweep an
# internal control that must reproduce the clean baseline exactly.
CORRUPTIONS = {
    "jitter": (jitter, [0.0, 0.005, 0.01, 0.02, 0.04, 0.08], "sigma / bbox diagonal"),
    "joint_dropout": (joint_dropout, [0, 1, 2, 3, 4, 6], "joints removed"),
    "limb_foreshorten": (limb_foreshorten, [1.0, 0.7, 0.5, 0.3, 0.1], "limb scale"),
    "hip_axis_collapse": (hip_axis_collapse, [1.0, 0.6, 0.3, 0.15, 0.05], "hip width scale"),
    "torso_axis_collapse": (torso_axis_collapse, [1.0, 0.6, 0.3, 0.15, 0.05], "torso height scale"),
}


def is_identity_severity(name, severity):
    """True when this severity is the operator's no-op control level."""
    return severity == CORRUPTIONS[name][1][0]


def apply(name, kpts, scores, severity, seed=0):
    """
    Apply a named corruption at a given severity.

    Args:
        name: key into CORRUPTIONS
        kpts: (17, 2) COCO-17 keypoints in pixel coordinates
        scores: (17,) COCO-17 confidences
        severity: one of CORRUPTIONS[name][1]
        seed: RNG seed; the caller derives it from (frame, operator, severity)
              so the whole sweep is reproducible.

    Returns:
        (kpts', scores') both float32 copies; inputs are never mutated.
    """
    if name not in CORRUPTIONS:
        raise KeyError(f"unknown corruption {name!r}; have {sorted(CORRUPTIONS)}")
    fn = CORRUPTIONS[name][0]
    rng = np.random.default_rng(seed)
    k_out, s_out = fn(
        np.asarray(kpts, dtype=np.float32),
        np.asarray(scores, dtype=np.float32),
        severity,
        rng,
    )
    return k_out.astype(np.float32), s_out.astype(np.float32)
