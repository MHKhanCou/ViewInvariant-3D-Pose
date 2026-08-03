# Anticipated Defense Questions & Answers

Numbers marked `[FREEZE]` are filled from frozen artifacts after Day 3.
Every answer's evidence file is named so it can be pulled up live.

## Protocol & validity

**Q: Your earlier numbers were invalidated. Why should we trust these?**
A: Three specific flaws were found and each is now structurally impossible:
(1) display post-processing in quantitative paths — the evaluation reads only
raw root-relative model output (`evaluation/lifting.py`, no camera_to_world);
(2) one frame repeated 27x — the corrected protocol lifts true 27-frame
detection windows, and only fully-windowed frames enter any reported number
(`evaluation/protocol.py: evaluated_centers`); (3) tuning and reporting on the
same data — the development pair (S1/Seq1 cam0-1) is labeled in every table
and all other 27 pairs plus subject S2 are held out.
Evidence: `thesis_artifacts/cross_view_eval/results_multicam.json`.

**Q: Is 28.4% still the number?**
A: 28.4% is the legacy single-frame-protocol figure on the dev pair; it
reproduces exactly (re-run 2026-08-03). The corrected protocol replaces it as
the primary claim: dev pair +20.5%, 27 held-out pairs mean +32.4%, held-out
subject S2 +13.4%. Both are reported, clearly labeled; the corrected numbers
are primary. Held-out exceeding dev is the anti-overfit evidence.

**Q: Why is this not MPJPE?**
A: MPJPE compares prediction to ground truth. Our cross-view metric compares
two predictions to each other, so we call it cross-view joint distance.
Where we do compare against ground truth (`evaluation/gt_eval.py`), we use
similarity-aligned per-joint distance and never label it MPJPE either, because
the alignment (Umeyama) differs from the H36M protocols.

**Q: Both cameras could be wrong identically — cross-view consistency would
not see it. How do you know your metric means anything?**
A: We anchored it against ground truth: Spearman rho between per-frame
canonical cross-view distance and actual GT error is +0.601 (p~1e-154,
n=1566) vs +0.188 for raw distance
(`thesis_artifacts/gt_validation/gt_results.json`). Consistency is a
necessary-not-sufficient signal, and we say so; the GT correlation shows it
carries real error information in practice — and canonicalization
strengthens it 3x.

## Reliability & abstention

**Q: Your abstention never fires on real data. Isn't the reliability score
useless?**
A: On clean studio footage it should not fire — that data is in-distribution
(clean-baseline reliability 0.872, abstention 0%). Under controlled
degradation the score tracks induced error with pooled Spearman rho = -0.813
(p~0, n=1760), joint dropout is abstained 100% of the time, and hard gates
fire on 1.9% of corrupted conditions while never firing on clean frames
(`thesis_artifacts/degradation/analysis.json`). The score separates clean
from corrupted; we deliberately did NOT tune the threshold to force
abstentions on clean data — that would be threshold-hacking.

**Q: Why geometric mean for combining components?**
A: Any single catastrophic component (e.g. detector confidence 0 after joint
dropout) should sink the score regardless of the others; an arithmetic mean
would let five healthy components mask one fatal one. The 100% abstention
under joint dropout is this property working.

**Q: Why not learn the reliability estimator?**
A: Khanal & Zhou (2026) show learned OOD detectors fail exactly on the
distribution shifts they were not trained on. A training-free estimator has
no training distribution to leave. That is the thesis position: reliability
signals derived from geometry (axis conditioning, symmetry, bone ratios)
transfer because they are properties of the skeleton, not of a dataset.

## Method scope

**Q: Why not a learned canonicalizer (3DPCNet)?**
A: Different goal. 3DPCNet learns SO(3) canonicalization with training data;
ours is a deterministic Gram-Schmidt construction requiring zero training,
running in O(17) per frame, and failing loudly (degenerate-axis gates) rather
than silently. The trade-off is honest: we do not claim to beat learned
methods on accuracy; we claim comparability without training.

**Q: Is this model-agnostic?**
A: Untested claim — we avoid it. It is model-independent by construction
(consumes any 17x3 root-relative pose), but only MotionAGFormer-XS was
evaluated. One model, stated plainly.

**Q: Your retrieval experiment got worse with canonicalization. Why include a
negative result?**
A: Because it delimits scope honestly. Canonicalization removes view
information by construction; a task whose signal partly IS view information
(retrieval across similar standing poses) loses discriminative power. This is
evidence the method does what it says — removes view — and defines where it
should not be used.

**Q: What does multi-scale add?**
A: Gate passed on every pair: dev +37.1%, held-out mean +36.4%, S2 +34.9%
over global canonicalization (`multiscale_results.json`). Anticipating the
"you just removed DOF and shrank the metric" objection: the multi-scale
distance retains full error information — rho(multi-scale dist, GT error)
= +0.610, matching global canonical's +0.601 — so the reduction is removed
limb-orientation variance, not metric deflation.

## Data & generalization

**Q: 1 subject, 1 sequence was your entire evaluation. Now?**
A: Corrected evaluation: 8 cameras -> 28 pairs on S1/Seq1 (1 dev + 27
held-out) plus a held-out subject S2/Seq1 pair, 54 fully-windowed frames per
pair, GT-anchored. Still one dataset and two subjects — scoped in
limitations; but the single-pair criticism is closed.

**Q: Why only S1 and S2?**
A: Those are the subjects present on local disk with annotations and
calibration. No new data was downloaded; all evaluation derives from
already-local AVIs, annot.mat, and calibration files.

**Q: The GT annot3 is in camera coordinates. Your predictions are in a
model-normalized space. How is the comparison valid?**
A: Per-frame similarity (Umeyama) alignment — rotation, translation, scale —
before measuring distance. Convention differences are absorbed by the
alignment; what remains is shape error. The GT bridge itself is validated:
world-transformed GT from different cameras agrees to 0.0 mm
(`gt_results.json: sanity`).

## Numbers audit trail

| Claim | File |
|---|---|
| Legacy 28.4% (dev pair) | `thesis_artifacts/cross_view_eval/results.json` |
| Corrected multicam table | `thesis_artifacts/cross_view_eval/results_multicam.json` |
| Reliability-vs-error, abstention under degradation | `thesis_artifacts/degradation/analysis.json` |
| GT correlations + bridge sanity | `thesis_artifacts/gt_validation/gt_results.json` |
| Ablation 3 conditions | `thesis_artifacts/coverage_error/ablation_results.json` |
| H36M baseline reproduction (45.1mm) | `thesis_artifacts/baseline_results.json` |
