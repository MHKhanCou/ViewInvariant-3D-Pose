# Pre-registration: multi-landmark frame estimation

Written and committed **before** the experiment is run, so git history shows the
criterion predates the number. Same practice as
`thesis_artifacts/tta/PREREGISTRATION.md`.

Date: 2026-08-05

## What is being tested

Not a new algorithm. Least-squares body-frame fitting from shoulders, hips and
ankles is established in the biomechanics literature and is cited as such. What
is being tested is whether **our own axis-length principle generalises from limb
frames to the global frame.**

The principle, established in Section 5.12 and replicated across two backbones
(+55.1% and +54.9%): the cross-view consistency of a frame is governed by the
length of the axis it is built from. It was discovered on *limb* frames. The
global frame still uses two vectors, and its lateral vector is the hip axis,
which is the shorter of the two lateral options available.

Measured on 2672 real poses:

| axis | mean length |
|---|---|
| torso, `P8 - P0` | 532.7 mm |
| shoulder, `P14 - P11` | 295.2 mm |
| hip, `P1 - P4` | 271.6 mm |

`cos(hip axis, shoulder axis) = +0.969`, same sign on 100% of poses, so the two
lateral estimates can be averaged without cancelling.

## Variants

Weighting is derived, not tuned. If per-joint noise is fixed and angular error
scales as 1/L, variance scales as 1/L², so inverse-variance weighting gives
`w = L²`. No hyperparameter is introduced.

- `both` — the current frame. `y = P8-P0`, `x = P1-P4`. Baseline.
- `hip_only` / `shoulder_only` — ablation isolating each lateral estimate.
- `weighted` — vertical from an L²-weighted mean of the unit vectors
  `P0→P7`, `P7→P8`, `P8→P9`, `P0→P8`; lateral from an L²-weighted mean of
  `P1-P4` and `P14-P11`.
- `svd` — vertical from the principal direction of the spine chain
  `{0, 7, 8, 9, 10}`; lateral as in `weighted`.

## Prediction

The axis-length principle predicts:

1. `weighted` and `svd` reduce mean canonical cross-view distance below `both`
   on the 180 held-out Human3.6M pairs.
2. `shoulder_only` beats `hip_only`, since the shoulder axis is 9 percent longer.
3. The ordering is the same on both backbones.

## Pass criterion

`weighted` or `svd` beats `both`, with a cluster bootstrap over subject-action
groups on the paired difference whose 95 percent interval excludes zero, on
**both** backbones.

## If it fails

Reported as a bound on the principle: axis length governs frames built from a
single segment, but combining anatomically distinct segments does not help,
plausibly because their errors are correlated through the shared pose estimate
rather than independent. That is a real finding and it constrains the
generalisation we would otherwise be tempted to state.

## Isolation

No file that produces an audited number is modified. The estimator lives in a new
module and the evaluation imports existing helpers without editing them.
