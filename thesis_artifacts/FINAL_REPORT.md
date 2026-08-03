# FINAL REPORT

Training-free, reliability-aware geometric canonicalization for cross-view
comparability of frozen monocular 3D pose predictions.

All numbers below are traceable to JSON artifacts in `thesis_artifacts/`
(file cited per claim). Predictions derive from frozen MotionAGFormer-XS;
no model was trained or fine-tuned. Frames: MPI-INF-3DHP S1/Seq1 (8 cameras)
and S2/Seq1 (2 cameras), extracted from local AVIs; hardware-synchronized
frame indices verified against `annot.mat` 2D overlays.

## Protocol

- **True 27-frame temporal windows.** Every reported frame is lifted from 27
  consecutive real detections; only fully-windowed centers (frames 13–66)
  enter any number. This corrects the single-frame-repeated-27× flaw in the
  earlier protocol.
- **Development pair labeled.** All historical tuning used S1/Seq1 cam0-cam1.
  It is marked in every table; the other 27 S1 pairs and subject S2 are
  held out.
- **Raw outputs only.** No display transforms (`camera_to_world`, z-shift,
  scaling) in any quantitative path.
- The metric is **cross-view joint distance** (prediction vs prediction),
  never MPJPE.

## Key Results

### 1. Cross-view consistency (corrected protocol)

`cross_view_eval/results_multicam.json`

| Scope | n pairs | Raw → Canonical improvement |
|---|---|---|
| Dev pair (S1 cam0-1) | 1 | +20.5% (0.1439 → 0.1143) |
| Held-out pairs (S1) | 27 | **mean +32.4%** (min −6.1%, max +59.1%) |
| Held-out subject (S2 cam0-1) | 1 | +13.4% (0.0858 → 0.0743) |

Held-out improvement exceeds the dev pair — no tuning-pair overfit.
The two negative pairs (cam1-cam2 −5.4%, cam2-cam4 −6.1%) have the smallest
raw distances (0.10, 0.06): near-aligned cameras leave little orientation
variance to remove, so canonicalization noise dominates.

Legacy reference: the earlier single-frame protocol gave 28.4%
(0.1172 → 0.0839) on the dev pair (`cross_view_eval/results.json`,
reproduced 2026-08-03). The corrected protocol supersedes it.

### 2. Procrustes oracle baseline

`cross_view_eval/results_multicam.json` (`oracle_cross_view_distance`)

Per-frame optimal rigid alignment lower-bounds any rotation-based
normalization. Canonical distance approaches the oracle on many pairs
(e.g. cam5-cam8: canonical 0.0457 vs oracle 0.0418; cam7-cam8: 0.0649 vs
0.0581) — the training-free frame recovers most of the recoverable
orientation variance on those pairs.

### 3. Ground-truth anchoring

`gt_validation/gt_results.json`

- GT bridge validity: world-transformed GT from different cameras agrees to
  **0.0 mm** (shape), verifying calibration handling.
- **ρ(canonical cross-view distance, GT error) = +0.601** (p ≈ 1e-154,
  n = 1566) vs **+0.188** for raw distance: canonicalization makes cross-view
  consistency a ~3× stronger rank-proxy for actual prediction error.
- ρ(reliability, GT error) pooled = −0.241 (p = 1.5e-8) on clean data;
  per-camera values vary (−0.75 to +0.71) because per-camera error variance
  is small on clean studio footage (restriction of range).
- GT error is similarity-aligned (Umeyama) per-joint distance in mm — not
  MPJPE (different alignment protocol).

### 4. Reliability under controlled degradation

`degradation/analysis.json` (2240 sweep records, identity controls exact)

- **Pooled ρ(reliability, induced canonical drift) = −0.813** (p ≈ 0,
  n = 1760): the training-free score ranks corrupted lifts by actual
  induced error.
- Joint dropout: **100% abstention** (detector-confidence component).
- Clean baseline: reliability 0.872, **0% abstention** — the score separates
  clean from corrupted without threshold tuning.
- Hard gates: 43/2240 corrupted conditions (1.9%), 0 on clean frames.

### 5. Reliability-aware ablation (Condition 3, real error axis)

`coverage_error/ablation_results.json`

Dev pair: raw 0.1439 → canonical 0.1143 (+20.5%) → reliability-filtered
canonical 0.1044 at 89% coverage (t=0.5), 0.0904 at 69% coverage (t=0.8).
Abstention fires on real data under the corrected protocol (temporal
stability now has genuine dynamic range). Coverage is always reported
alongside error.

### 6. Multi-scale canonicalization (per-limb frames)

`cross_view_eval/multiscale_results.json`

Pre-declared gate: must beat global canonicalization on the dev pair.
**Gate passed on all 29 pairs**: dev +37.1%, held-out mean +36.4%
(+23.9% to +54.1%), held-out subject +34.9%. The multi-scale distance
retains error information — ρ(multi-scale distance, GT error) = +0.610,
matching global canonical (+0.601) — so the reduction is removed
orientation variance, not metric shrinkage.

### 7. Training-free multi-view selection and fusion

`fusion/fusion_results.json`

Because canonicalization expresses every camera in the same body frame,
predictions from N uncalibrated cameras become directly comparable — and
therefore selectable and fusable — with no extrinsics, no correspondence
search, and no training. Reference validity checked first: GT canonicalized
from different cameras agrees to **0.00 mm**, so the reference frame is
genuinely view-independent.

Error vs GT (similarity-aligned; Wilcoxon p = 1.6e-10 for every strategy
against the baseline) on two windows: the **static** window (frames 0–79,
low body rotation) and a **dynamic** window (frames 576–721, 138° net body
rotation) chosen by a ground-truth-only motion scan.

| Strategy | Static (54 f) | Dynamic (120 f) |
|---|---|---|
| Worst single view | 276.3 mm | — |
| **Arbitrary single view** (deployment baseline) | **148.7 mm** | **214.9 mm** |
| Reliability-weighted fusion | 113.5 mm (+23.7%) | 192.0 mm (+10.6%) |
| Median fusion | 101.0 mm (+32.1%) | 196.2 mm (+8.7%) |
| Reliability-*selected* view | 98.3 mm (+33.8%) | 211.7 mm (**+1.5%**) |
| Best view (oracle — requires GT) | 87.9 mm | 176.5 mm |

**Fusion is the supported contribution; selection is not.** Fusion helps in
both regimes because averaging suppresses independent per-view error without
needing to know which view is best. On the static window fusion gains scale
with camera count (2 views 130 mm → 8 views 113 mm) while the arbitrary-view
baseline stays flat; the held-out subject reproduces it (+8.2% against a
+10.8% oracle ceiling).

**Negative result — the reliability score does not rank views by accuracy.**
The apparent +33.8% from selection on the static window is an artifact, and
we report it as one. Three pieces of evidence:

1. On static footage the score picks the *same* camera in all 54 frames — it
   is not selecting per frame. A best-fixed-camera policy (which needs GT to
   choose the camera) scores 90.2 mm, better than the 98.3 mm "adaptive" pick.
2. On the dynamic window the score *does* switch (6 distinct cameras, 22%
   switch rate), but its choices are no better than chance: the picked view's
   true-error rank is **4.78 of 8** against a random expectation of 4.5.
3. Directly: the **within-frame** Spearman correlation between reliability and
   error across simultaneous views is ≈ 0 in both regimes (−0.112 static,
   −0.097 dynamic).

The static gain therefore came from a constant camera choice combined with a
large within-frame error spread (188 mm), which beats an average dragged down
by poor views — not from identifying good ones.

**What this delimits.** The score measures geometric *plausibility*. That
detects corruption strongly (ρ = −0.813 under induced degradation, §4; and it
switches away from a deliberately corrupted view in 100% of frames) but does
not detect viewpoint-induced depth error, because a pose can be perfectly
plausible — symmetric, correct bone ratios, well-conditioned axes — and still
be wrong in depth. Analytic geometric reliability is a corruption detector,
not an accuracy estimator. We consider this delimitation a finding in its own
right, and it is the reason the framework's abstention claim is scoped to
degradation rather than to general error prediction.

### 8. Negative result: retrieval

`retrieval/retrieval_results.json`

Canonicalization worsens cross-view retrieval (R@1 0.02 → 0.00, MRR −6.9%).
Expected: retrieval across similar standing poses relies partly on view
information, which canonicalization removes by construction. This delimits
scope: the framework targets cross-view comparison, not general-purpose
representation.

## Baseline integrity

`baseline_results.json` — frozen MotionAGFormer-XS H36M reproduction:
MPJPE 45.149 mm (official 45.1), P-MPJPE 36.892 mm (official 36.9).
(MPJPE terminology is correct here: this is the official GT benchmark.)

## Limitations

- One dataset (MPI-INF-3DHP), two subjects, one model (MotionAGFormer-XS).
  The framework is model-independent by construction (consumes any 17×3
  root-relative pose) but model-agnosticism is untested and not claimed.
- Reliability tracks error strongly under degradation but only moderately on
  clean data, where error variance is small.
- Near-aligned camera pairs gain little and can regress slightly.
- Abstention threshold (0.5) was frozen at implementation time, not
  calibrated on the reporting pairs.
- Multi-view selection is not per-frame adaptive on clean footage (see §7);
  adaptivity is demonstrated only under induced degradation. Fusion results
  come from one sequence of 54 synchronized frames.

## Invalidated figures (do not cite)

The earlier 67% and 40% reductions used display post-processing and a
repeated frame; they are invalid. The 28.4% legacy figure is valid but
superseded by the corrected protocol above.
