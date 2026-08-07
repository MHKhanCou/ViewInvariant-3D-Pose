# Pre-registration 14: 2D-Input Invariance of the Frozen Lifting Stage

**Status:** Pre-registered before any run. Commit of this file + the runner
predates the result artifact.

**Date:** 2026-08-07

## Question

Through the *real* detection + lifting path (YOLOv8 → frozen MotionAGFormer
lifter, byte-identical to the cached evaluation pipeline), does 2D keypoint
corruption create a 3D corruption regime? And does the real detector
confidence channel carry any usable signal on clean data?

This bounds the thesis's failure-surface map. Experiments 12–13 corrupt the
lifted **3D** predictions directly and show that the anatomical frame and the
Kabsch template fail on disjoint joint supports. If 2D keypoint corruption
also propagated, the routing rule would need a detector-side gate; if it does
not, the failure surface is confined to the 3D alignment level, and the
routing rule's gate is correctly modelled at that level.

## Protocol

- **Pipeline:** `backend.model_loader.get_detector()` + `get_model()`,
  `evaluation.lifting.lift_from_coco_window` — identical code path to
  `evaluation.run_eval`; sanity anchor: clean re-lift must reproduce the
  cached `predictions_cache.npz` within 1.0 mm mean per-joint error.
- **Data:** MPI-INF-3DHP S1/Seq1, static cameras 0 and 1, all available
  frames. Frames are re-detected (real YOLOv8), so the measured confidence
  channel is genuine, not simulated.
- **Corruption:** the 2D keypoints of a joint group are displaced by a fixed
  vector (seeded random direction per joint, magnitude `f * bbox_diag`) in
  every frame of the lifting window; keypoint scores are left at their
  detected values (the corruption is *confidently wrong*, not missing).
  Groups: **distal** (COCO 7,8,9,10,13,14,15,16: elbows, wrists, knees,
  ankles), **core** (COCO 5,6,11,12: shoulders, hips).
  Magnitudes: `f ∈ {0.03, 0.10, 0.15}` of the bbox diagonal (≈ 87–434 px on a
  2048 px frame; the largest is 60% of the normalized input space at the
  wrist).
- **Primary metric:** mean per-joint `|Δ|` (mm) between the corrupted and
  clean lifted 3D pose, averaged over frames, per (group, magnitude). Also
  recorded: per-group delta and per-joint max.
- **Detector channel:** distribution of per-frame mean keypoint confidence on
  the clean re-detection (min, mean, fraction < 0.9).
- **Reliability channel (context):** the cached analytic reliability score for
  the same cameras, reported for context (it is the measured signal the
  abstention mechanism already uses).

## Predictions

- **P1 (invariance):** at the largest magnitude `f = 0.15`, the mean per-joint
  `|Δ|` is **< 3.0 mm** for both joint groups. (The frozen lifter absorbs 2D
  keypoint corruption; no 2D-side corruption regime exists.)
- **P2 (saturated confidence):** on clean frames the mean detector confidence
  is **≥ 0.99** and the fraction of frames with mean confidence < 0.9 is
  **≤ 5%**. (The real confidence channel carries no usable gate signal on
  clean data.)
- **P3 (context, no re-run):** the same pipeline under *3D* anchor corruption
  moves the pose ≥ 50 mm (Experiment 12 measured 53.45 → 337.87 mm). The
  failure surface is at the 3D alignment level.

## Reading

- **Reading 1 (boundary confirmed):** P1 and P2 hold. The 2D channel is inert;
  the failure map and the routing rule are correctly scoped to the 3D level.
- **Reading 2:** P1 holds, P2 fails. The confidence channel varies on clean
  data — the simulated-gate caveat in Experiment 13 must be re-opened.
- **Reading 3:** P1 fails (displacement propagates ≥ 3.0 mm). The 2D channel
  creates a real corruption regime; a detector-side gate is required and the
  failure map extends to the 2D level.

## Honest boundaries

- This measures 2D **keypoint** corruption, not detection failure (missed
  person, truncated pose) or motion blur; those remain out of scope.
- Invariance is measured at the *lift output*; per-joint deltas of ~0.1–0.4 mm
  are within the numerical noise of the metric (the cached re-lift anchor is
  0.00 mm, so the comparison is not dominated by pipeline variance).
- The three probe runs performed during design (S1 cam0 slice, 24–80 centers,
  magnitudes 0.01–0.15) were exploratory; this artifact supersedes them as
  the formal record.
