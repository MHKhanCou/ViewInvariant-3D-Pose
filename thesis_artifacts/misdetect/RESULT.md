# Result 14: 2D-Input Invariance of the Frozen Lifting Stage

**Pre-registration:** `PREREGISTRATION.md` (commit `4ccee2f`), with the
documented P2 amendment (incorrect cache read corrected before finalizing).
**Date:** 2026-08-07 · **Verdict: Reading 1 — the 2D channel is inert; the
failure surface is confined to the 3D alignment level.**

## Protocol executed

MPI-INF-3DHP S1/Seq1, static cameras 0 and 1, all frames re-detected with the
real YOLOv8 `detect_with_rotation` path (byte-identical to the cached
evaluation pipeline), lifted with the frozen MotionAGFormer lifter.
Displacement magnitudes f ∈ {0.03, 0.10, 0.15} of the bbox diagonal on the
distal and core joint groups; keypoint scores left at detected values
(*confidently wrong*, not missing).

Sanity anchor: clean re-lift vs `predictions_cache.npz` = **0.00 mm** on both
cameras.

## Results

### P1 — 2D keypoint displacement does not propagate to the 3D lift

Mean per-joint `|Δ|` vs clean lift (mm), bootstrap 95% CI over frames:

| condition | cam0 | cam1 |
|---|---|---|
| distal f=0.03 (87 px) | 0.06 | 0.06 |
| distal f=0.10 (290 px) | 0.15 | 0.18 |
| distal f=0.15 (434 px) | 0.21 | 0.25 |
| core f=0.03 | 0.15 | 0.08 |
| core f=0.10 | 0.26 | 0.23 |
| core f=0.15 | **0.32** | **0.33** |

- Worst mean over both cameras at f=0.15: **0.33 mm** (threshold: 3.0 mm).
- Worst per-joint delta across all conditions: **0.77 mm** (cam0, core).
- The corruption reaches the model input: a 434 px wrist displacement is a
  **0.60-unit move in the normalized input space** (60%), yet the lifted
  output moves ≤ 0.4 mm per-joint.

### P2 (amended) — the real detector-confidence channel carries no usable gate signal

| camera | mean conf | min conf | frac < 0.9 |
|---|---|---|---|
| cam0 | 0.738 | 0.713 | **1.000** |
| cam1 | 0.812 | 0.806 | **1.000** |

Every frame in both cameras lies below 0.9: no confidence threshold can
separate frames on clean data, so a confidence gate would fire everywhere or
nowhere. (The original P2 ≥ 0.99 figure was an incorrect cache read —
`components[:, 0]` is a reliability component, not detector confidence; the
amendment in the pre-registration documents this.)

For context, the *measured* analytic reliability score (already used by the
abstention mechanism) **does** vary: cached reliability mean 0.752, min 0.0
(cam0) and mean 0.864, min 0.854 (cam1).

## Interpretation

1. **The 2D channel cannot create a 3D corruption regime.** Even a 434 px
   (21 % of a 2048 px frame, 60 % of normalized input space) displacement of
   the torso anchors moves the lifted pose by ≤ 0.33 mm. Contrast with
   Experiment 12, where corrupting the same joints at the **3D** level moves
   the pose **53.45 → 337.87 mm** — a ~1000× gap.
2. **The failure surface is at the 3D alignment level, end to end.** The
   frame-vs-template failure-support map (Experiments 12–13) is correctly
   scoped: real detection errors of the kind that reach the pipeline's output
   are 3D-level errors, and the routing rule's gate is rightly modelled at
   that level.
3. **A confidence gate is not deployable on this pipeline's clean signal.**
   The routing rule's confidence gate remains a *simulated* stand-in; the
   measured signal that actually varies is the analytic reliability score.
   This is an honest boundary of Experiment 13, now measured rather than
   assumed.

## Honest boundaries

- Measures 2D **keypoint** corruption only; detection failure (missed person,
  truncation, motion blur) is out of scope.
- Invariance is at the lift output; deltas of 0.06–0.33 mm are above the
  0.00 mm pipeline-noise anchor but far below any operational pose scale.
- The absolute level of the fresh detection scores (0.74–0.81) differs from
  the cached detector-confidence component (0.95–1.0); both are reported
  honestly, and the verdict's P2 does not depend on the absolute level, only
  on the (absent) variation.
- Secondary finding: the frozen lifter is also invariant to *zeroing* distal
  keypoints (probe: ≤ 0.2 mm at full-frame dropout), consistent with the
  displacement result.
