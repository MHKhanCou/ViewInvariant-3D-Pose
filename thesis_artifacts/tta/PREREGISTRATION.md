# Pre-registration — TTA dispersion as an error predictor

**Committed before the experiment was run.** Git history is the evidence that this
criterion predates the number. If the result fails these thresholds it is reported as a
falsification, not re-specified.

## Question

Does the disagreement a frozen monocular 3D pose model already produces across test-time
augmentations predict its own error, using a **single** camera?

Motivation: the 6-component `reliability` score has been falsified four independent ways
(within-frame ρ≈0 across simultaneous views; view selection straddling chance at 4.78/8
on S1 and 3.67/8 on S2; sign inversion across backbones, −0.707 MotionBERT vs +0.375 MLP;
reliability-weighted model fusion worse than a plain mean). The only signal that works —
cross-view canonical disagreement, ρ = +0.601 — needs two cameras.

## Incumbent bar

From `thesis_artifacts/gt_validation/gt_results.json`:

| Quantity | Value |
|---|---|
| pooled ρ(reliability, GT error) | **−0.241** (n = 540) |
| pooled ρ(canonical cross-view distance, GT error) | **+0.601** |

0.241 is the floor to beat; 0.601 is the two-camera reference ceiling.

## Primary predictor (fixed in advance)

`disp_procrustes` — mean pairwise `similarity_align_error` over the **K = 6** primary
arms: 2 flip branches × 3 keypoint-jitter levels (σ ∈ {0, 0.005, 0.01} of bbox diagonal).

K is held **constant per frame**. Mean pairwise dispersion is biased by K, so a varying
arm count would partly measure how many arms survived rather than model disagreement.

Chosen because it is the *identical* metric used to compute GT error, making the
comparison against `reliability` maximally clean.

**Known property, found by unit test before the run and recorded here rather than
silently fixed:** `disp_procrustes` is invariant to a global rotation/translation of all
arms but **equivariant to a global scale** — it reports an absolute distance in the poses'
own units. So does the GT error it is compared against. This means a frame whose predicted
skeleton is simply *larger* has both a larger dispersion and a larger absolute error, which
could inflate ρ without any genuine predictive content. The primary predictor is **not**
changed post-hoc (that would break pre-registration), but `disp_scale` is reported
alongside it, and if the correlation is driven by scale it will show up as `disp_scale`
correlating with error at a similar magnitude. Read the two together.

## Pass criterion — all three required

- **(a) Usefulness floor.** Pooled ρ(disp, GT error) ≥ **+0.30**. Deliberately stricter
  than merely beating the already-falsified −0.241: a signal that only beats a broken one
  is not worth shipping.
- **(b) Stability.** Same sign across all three strata: S1 static, S1 dynamic, S2 dynamic.
  Sign instability is precisely how the view-selection claim died (4.78/8 vs 3.67/8) and
  how the backbone transfer failed (−0.707 vs +0.375).
- **(c) Significance over the incumbent.** Paired Δ|ρ| against `reliability` on identical
  frames, 95% CI excluding 0 under a **cluster bootstrap resampling cameras**
  (10 000 draws, seed 12345). Frames within a camera share overlapping 27-frame windows,
  so an i.i.d. frame bootstrap would be anticonservative; the camera is the independent
  unit.

## Reported regardless of outcome

**Partial Spearman ρ(disp, error | detector_confidence)**, controlling on cached
`components[:, 4]`. If the signal vanishes under this control, TTA dispersion is a
detector-confidence proxy and contributes nothing new — and that finding ships as the
result.

## Exploratory (flagged `"exploratory": true`, no claims)

`disp_depth` / `disp_inplane` (camera-z vs in-plane decomposition — the mechanism test),
`disp_scale`, per-joint canonical dispersion vectors, and the rotation arm (K = 8).

The mechanism test is run in **camera coordinates**, not body coordinates: the hypothesis
concerns depth error, and `canonicalize_single`'s z-axis is the body sagittal axis, not
camera depth.

## Scope limits stated in advance

- Canonicalization is **not** claimed as the enabler. Verified against source: augmented
  predictions already return in the same camera frame — `detect_with_rotation` remaps
  keypoints back (`pose_detector.py:120-135`), `lift_from_coco_window` un-mirrors the flip
  branch via `flip_data` before averaging (`lifting.py:94`), and `gt_eval.py:144` scores
  `raw`, not `canonical`. Canonicalization is credited only for making per-joint
  dispersion vectors comparable across cameras/subjects.
- The rotation arm is secondary and reported only on frames where all four angles detected
  across the full 27-frame window. Prior work (`scripts/rotation_robustness_test.py`)
  found YOLO failing at 90° (8 spurious detections, all below the 0.4 gate), so that arm
  can measure detector failure rather than model disagreement. Drop rates are reported.
- One dataset (MPI-INF-3DHP), one backbone (MotionAGFormer-XS), 2 subjects.

## Outcome handling

**Pass** → claim: *a frozen single-view model's disagreement across free test-time
augmentations predicts its own error*. Single-view, zero added parameters, zero labels,
near-zero extra compute.

**Fail** → written up as **falsification axis 5**: five independent tests of a
training-free quality signal for frozen 3D pose, all reported, alongside the one signal
that does work (cross-view, ρ = +0.601) and the calibration-free fusion it enables.
