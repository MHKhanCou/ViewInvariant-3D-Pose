# Pre-registration: the axis-length law at the joint level

Written and committed **before** the experiment is run. Same practice as
`thesis_artifacts/tta/`, `thesis_artifacts/multilandmark/` and
`thesis_artifacts/conditioning/`.

Date: 2026-08-05

## What is new here

Section 3.2 derives `theta_rms ~= 2*sigma/L` and Section 5.16 tests the law
**between frame constructions**, where L spans 138 to 461 mm. The conditioning
experiment then showed it does **not** discriminate **between frames of one
construction**, because there L is an anatomical near-constant.

Neither test touches the level the derivation actually speaks to most directly:
**within a single canonicalized pose, across joints.** That prediction has not
been made or tested anywhere in this report, and it needs no new data.

## The prediction, derived

Canonicalization applies the estimated frame. If the frame carries a small
rotation error with rotation vector `Theta`, a joint at position `p_j` is
displaced by `delta_j = Theta x p_j`, so

    ||delta_j|| = ||Theta|| * r_j * sin(phi)

with `r_j` the joint's radius from the root. Averaging `sin^2(phi) = 2/3` over
isotropic orientations, and taking two views whose frame errors are independent,
the frame-induced part of the cross-view distance at joint j is

    d_frame(j) = sqrt(4/3) * ||Theta|| * r_j                     ... (linear in r_j)

The frame-induced part is separable from the estimator's own error, because a
per-frame Procrustes rotation removes **exactly** the rigid misalignment and
nothing else:

    d_frame(j) ~= sqrt( d_canonical(j)^2 - d_oracle(j)^2 )

## The constants come from other experiments, so nothing is fitted

    sigma    = 7.5 mm     fitted on the eight limb frames (Section 5.16)
    L_hip    = 275.8 mm   measured over 52900 frames (conditioning pre-reg)
    L_torso  = 456.1 mm   measured, multiscale artifact

The Gram-Schmidt frame takes its tilt from the torso axis and its roll from the
hip axis, so `||Theta||` lies between the torso-only and the both-axes values:

    lower   2*sigma/L_torso                              = 0.0329 rad
    upper   sqrt((2s/L_hip)^2 + (2s/L_torso)^2)          = 0.0636 rad

Multiplying by `sqrt(4/3) = 1.1547` gives a **predicted slope band of
[0.038, 0.073]**, with nothing fitted to the data being tested.

## Known bias, and which way it points

Procrustes fits the rotation that minimises total squared error, so it absorbs
some genuine shape disagreement as well as frame error. `d_oracle` is therefore
too small, and `d_canonical^2 - d_oracle^2` is too large. **The measured slope is
biased upward.** A slope below the band is stronger evidence than one above it,
and this is stated now rather than after seeing the number.

## Predictions

1. **Linear in radius (primary).** Regressing `sqrt(d_can^2 - d_orc^2)` on `r_j`
   over the seventeen joints gives `R^2 >= 0.80` on both backbones.
2. **Slope in the predicted band (secondary).** The fitted slope falls in
   [0.038, 0.073]. Reported either way; see the bias note above.
3. **Not a restatement of estimator error (the control).** Distal joints are
   harder to estimate, so raw per-joint error also grows with radius. The oracle
   distance `d_oracle(j)`, which has all frame error removed, must have a
   materially smaller slope against `r_j` than `d_canonical(j)` does - we require
   at least a factor of two. If the two slopes are comparable, the decomposition
   is measuring the estimator rather than the frame, and the result is void.

## Pass criterion

Predictions 1 and 3 must hold **on both backbones**. Prediction 2 is reported
either way, because the theory supports the structure more strongly than the
constant, which is the lesson of Section 5.16 where the mechanism replicated and
the functional form did not.

## If it fails

Reported as a bound, like the other three. Failure of 1 would say the frame error
is not a small rotation in the sense the derivation assumes. Failure of 3 would
say the Procrustes decomposition does not separate the two error sources, which
would also weaken the oracle's use as a floor elsewhere in the report and would
be worth stating for that reason alone.

## Isolation

New module, new artifact directory. Imports existing helpers and edits none of
them. No file producing an audited number is touched.
