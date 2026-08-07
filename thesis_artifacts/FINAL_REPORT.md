# FINAL REPORT

Training-free geometric canonicalization for cross-view comparability of frozen
monocular 3D pose predictions.

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

| Strategy | S1 static (54 f) | S1 dynamic (120 f) | S2 dynamic, held-out subject (120 f) |
|---|---|---|---|
| **Arbitrary single view** (deployment baseline) | **148.7 mm** | **214.9 mm** | **212.1 mm** |
| Reliability-weighted fusion | 113.5 mm (+23.7%) | 192.0 mm (+10.6%) | 190.5 mm (+10.2%) |
| Median fusion | 101.0 mm (+32.1%) | 196.2 mm (+8.7%) | 191.4 mm (+9.8%) |
| Reliability-*selected* view | 98.3 mm (+33.8%) | 211.7 mm (+1.5%) | 199.5 mm (+6.0%) |
| Best view (oracle — requires GT) | 87.9 mm | 176.5 mm | 171.9 mm |

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
2. On dynamic windows the score *does* switch (S1: 6 cameras, 22% switch rate;
   S2: all 8 cameras, 24%), but the quality of its choices **straddles chance**
   across the two sequences: the picked view's true-error rank is **4.78 of 8**
   on S1 (random expectation 4.5, i.e. slightly worse than random) and **3.67**
   on S2 (slightly better). A signal that lands on both sides of chance on two
   sequences of the same protocol is not a usable ranking signal.
3. Directly: the **within-frame** Spearman correlation between reliability and
   error across simultaneous views is ≈ 0 in both regimes (−0.112 static,
   −0.097 dynamic).

By contrast, **fusion replicates tightly across subjects and regimes**
(+23.7% / +10.6% / +10.2% / +8.2%), which is what a real effect looks like.

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

### 8. A single-view training-free error predictor that works

`bone_consistency/bone_consistency.json`

Sections 4 and 7 establish that *within-frame geometric plausibility* cannot see
depth error. Section 3 shows that *cross-view disagreement* can (ρ = +0.601) —
but needs two cameras. This section closes the gap with a **single-camera**
signal built on a physical invariant: **a person's bones do not change length**,
so any temporal variation in predicted bone lengths is direct evidence of error,
and monocular depth ambiguity produces exactly that through foreshortening.

Signal, scale-free by construction: per frame, divide the 16 bone lengths by that
frame's mean bone length (removing global scale), estimate the subject's own
skeleton as the median ratio vector over a reference window (no labels, no
ground truth), then measure each frame's relative deviation from it.

| Quantity | Value |
|---|---|
| **ρ(bone deviation, GT error)** | **+0.492** (n = 2460, 26 camera-streams) |
| ρ(reliability, GT error) — incumbent | −0.192 |
| Partial ρ controlling for detector confidence | **+0.481** |
| Causal estimate (reference window disjoint from evaluated frames) | **+0.473** (n = 1230) |
| Cluster bootstrap 95% CI on \|ρ_bone\| − \|ρ_reliability\| | **[+0.108, +0.515]** |
| Per stratum (S1 static / dynamic, S2 static / dynamic) | +0.428 / +0.167 / +0.609 / +0.222 |

**Every stratum agrees in sign** — the test that retired the view-selection claim
(§7), where the two dynamic sequences landed on opposite sides of chance.

Robustness beyond the required checks: the signal is not a detector-confidence
proxy (partial ρ barely moves); it is not a restatement of scale (the scale-free
ratio form scores +0.492 versus +0.367 for pure scale deviation); and it is not
transductive — a **causal** reference window scores +0.473, and a skeleton
estimated from a *different camera* still scores +0.440, as it should, since it
is the same person.

**Honesty note.** This signal was found by exploratory probing, not
pre-registered. It is therefore held to the same three criteria pre-registered
for the TTA experiment (`thesis_artifacts/tta/PREREGISTRATION.md`, committed
before that run), plus the two robustness checks above — five in total, all
passed. It should be replicated on a second dataset before it carries weight in
a submission.

**Scope limit — frame-level, not joint-level.** The signal predicts *which
frames* to distrust, not *which joints within a frame*. Assigning each joint the
mean deviation of the bones touching it gives only ρ = +0.167 pooled over 41 820
joint-frames, and **within-frame joint ranking is chance**: mean ρ = −0.010,
positive in 49% of frames. This mirrors the reliability score's limitation
exactly (§7): these geometric signals discriminate *between* poses, not *inside*
one. Claim frame-level triage only; spatial error localization remains open.

#### 8a. Why it works where test-time augmentation does not

`tta/tta_results.json` — pre-registered, **FAILED all three criteria**

We also tested whether the model's *self-disagreement* predicts its error, by
harvesting the augmented predictions the pipeline already computes and discards
(4 rotated detections, of which 3 are dropped; flipped and unflipped lifts,
which are averaged). Dispersion over K = 6 fixed arms gives:

| Criterion | Required | Observed |
|---|---|---|
| pooled ρ | ≥ +0.30 | **+0.100** |
| same sign across strata | yes | **no** (−0.175, +0.413, −0.088, +0.278) |
| bootstrap CI excludes 0 | yes | **no** ([−0.592, +0.476]) |
| partial ρ given detector confidence | reported | **−0.083** (collapses) |

The contrast is the finding: **the model agrees with itself while being wrong.**
Perturbing the input does not move a confidently-wrong depth estimate, so
self-consistency carries almost no error information — consistent with Khanal &
Zhou (2026) on learned confidence failing off-distribution. A *physical
invariant* the prediction must satisfy (constant bone length) does carry that
information, because the world constrains it and the model does not.

#### 8b. The incumbent score's defect is its combination rule, not its parts

`bone_consistency/bone_consistency.json` → `component_analysis`

Four of the six components individually predict error roughly twice as well as
their geometric mean: temporal_stability −0.362, bilateral_symmetry −0.350,
abnormal_bone_ratio −0.316, torso_hip_angle −0.312, versus the composite's
−0.192. The mean is diluted by axis_conditioning (−0.092, and the only component
without a consistent sign across strata) and detector_confidence (−0.060).

Selecting components on subject **S1 only** and scoring on held-out subject
**S2** (so the selection cannot be curve-fitting):

| Signal | S1 (selection) | S2 (held-out) |
|---|---|---|
| Incumbent composite, all six | −0.222 | −0.162 |
| Pruned composite (4 components) | −0.509 | **−0.357** |
| Bone deviation alone | +0.572 | +0.368 |
| **Bone deviation + pruned composite** | +0.561 | **+0.395** |

Pruning more than doubles held-out performance, and the two signals combine to
beat either alone — they are complementary, one measuring within-frame
plausibility and the other a cross-time physical invariant.

### 9. Negative result: retrieval

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

## What "training-free" does and does not mean here

The phrase is attackable and must be used precisely. **The pipeline is not
training-free — the backbone and detector are both trained.** What is
training-free is everything this work adds on top of them.

| Element | Trained parameters | Labels used | Status |
|---|---|---|---|
| MotionAGFormer-XS backbone | **yes** (H36M, by its authors) | yes | frozen, not ours |
| YOLOv8-pose detector | **yes** (by its authors) | yes | frozen, not ours |
| Gram-Schmidt canonicalization | **0** — closed form | none | training-free |
| Multi-scale per-limb frames | **0** — closed form | none | training-free |
| Multi-view fusion | **0** — weighted average | none | training-free |
| Reliability score | **0 learned** — but ~6 hand-set constants (0.1×, 3×, 0.3, 2.5 bone thresholds; 0.5 abstention) | none | training-free, **not tuning-free** |
| Bone-length inconsistency (§8) | **0**; per-subject median estimated at test time from unlabeled predictions | **none** | training-free and label-free |
| **Pruned composite (§8b)** | **0** | **YES — ground-truth error on S1 selects 4 of 6 components** | **NOT label-free** |

Precise claims, then:

- The framework **adds zero learned parameters** to a frozen pipeline. That is
  checkable by reading the code and is what distinguishes it from 3DPCNet, which
  is also estimator-agnostic but *trains* a canonicalization network.
- The bone-length signal is additionally **label-free**: the subject's skeleton
  is the median of the model's own unlabeled predictions, computed at test time.
  It is adaptive (per-subject) but not learned — no parameters cross subjects.
- The **pruned composite is not label-free** and must not be presented as such.
  Selecting 4 of 6 components consumed ground-truth error on S1. It is reported
  as an *analysis of why the incumbent underperforms*, not as a deployable
  training-free method. A label-free selection rule (e.g. dropping components
  whose sign is unstable across cameras, which would also have dropped
  `axis_conditioning`) is future work and was not tested here.
- Hand-set constants are not learned but they were chosen by a human, so
  "training-free" never implies "no free parameters".

## Positioning against prior work

A literature sweep (2026-08-04) requires two claims to be withdrawn:

- **Canonicalization is not claimed as novel or more accurate.** 3DPCNet
  (arXiv 2509.23455, 2025) already performs estimator-agnostic post-hoc
  canonicalization of frozen 3D predictors, and reports a hand-built
  anatomical-landmark baseline of the same family as ours losing to it
  (62.9–64.6 mm / 20.6–21.6° vs 47.6 mm / 3.4°). Canonicalization here is
  **substrate**, not contribution.
- **The hip/spine body frame is established preprocessing**, e.g. V-VIPE
  (CVPR-W 2024) aligns hips and spine analytically via Kabsch. Pr-VIPE staked
  calibration-free cross-view comparability in 2019.

What remains defensible is a **requirement profile**: every canonicalization
competitor requires training (MoViD: GT SMPL supervision; V-VIPE: 3D-GT VAE;
3DPCNet: self-supervised on synthetic rotations), and every post-hoc
uncertainty competitor requires a **labeled calibration set** (conformal
keypoint detection CVPR 2023, CHAMP, CUPS). This framework requires no
training, no labels, and no camera parameters — and offers **no coverage
guarantee**, which we state rather than imply.

We avoid "first to" phrasing entirely: the view-selection and part-based
limb-frame literatures have not been systematically searched.

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
