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

**Q: 3DPCNet (2025) already does post-hoc canonicalization of frozen
estimators — and reports a hand-built anatomical baseline like yours losing
to it badly (62.9–64.6 mm vs 47.6 mm; 20.6–21.6° vs 3.4°). What is left of
your contribution?**
A: We accept that finding and do not contest it. Canonicalization is not our
claim — it is the substrate, and where a learned canonicalizer is available
and its training data acceptable, it is the better choice for accuracy. Our
contribution sits downstream: a *label-free* analytic reliability signal on a
frozen predictor, what it enables (calibration-free multi-view fusion), and
where it stops working. 3DPCNet has no reliability, no abstention, and no
multi-view component; it also requires self-supervised training on
synthetically rotated poses. The honest framing is a requirement profile
(no training, no labels, no camera parameters), not an accuracy win.

**Q: Then is anything in your method actually new?**
A: We claim a *combination* and a *delimitation*, not a primitive. Every
canonicalization competitor requires training (MoViD: GT SMPL; V-VIPE: 3D-GT
VAE; 3DPCNet: self-supervised); every post-hoc uncertainty competitor requires
a labeled calibration set (conformal keypoint detection, CHAMP, CUPS). Ours
requires neither. We deliberately avoid "first to" phrasing: we have not
systematically searched the view-selection and part-based normalization
literatures, so we state what our method requires, not who got there first.

**Q: Isn't the hip/shoulder body frame just standard preprocessing?**
A: Yes — V-VIPE does the same alignment via Kabsch and calls it preprocessing,
and we cite it as such. We do not claim it. What we add on top is the
multi-scale per-limb extension (every published anatomical baseline we found
uses a single global frame) and the reliability layer.

**Q: Why not a learned canonicalizer, then?**
A: For accuracy, use one. Ours applies where training is not an option:
frozen third-party predictors, no labels for the deployment distribution, no
camera calibration. That is a deployment constraint, not a claim of superiority.

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

## Multi-view selection & fusion

**Q: Is the view selection actually adaptive, or does it just always pick the
same camera?**
A: It is not adaptive, and we tested this rather than assuming it. On static
footage it picks the same camera in all 54 frames. On a dynamic window (138°
body rotation) it does switch — 6 cameras, 22% switch rate — but its choices
rank 4.78 of 8 by true error against a random expectation of 4.5, i.e. no
better than chance. We therefore do NOT claim view selection. Fusion is the
supported contribution.

**Q: Then what does the reliability score actually do?**
A: It measures geometric plausibility, which detects corruption but not
viewpoint-induced depth error. Under induced degradation it tracks error at
ρ = −0.813 and abstains on 100% of joint-dropout cases; across simultaneous
clean views its within-frame correlation with error is ≈ 0 (−0.112 static,
−0.097 dynamic). A pose can be perfectly plausible — symmetric, correct bone
ratios — and still be wrong in depth. We state this as a delimitation: it is
a corruption detector, not an accuracy estimator.

**Q: Your report shows a +33.8% selection gain. Is that real?**
A: It is real as a number and misleading as a claim, so we explain it. That
window has a 188 mm within-frame error spread, so picking one constant decent
camera beats an average dragged down by bad views. It reflects a lucky
constant, not ranking ability — which is why the dynamic window collapses it
to +1.5%. We report both and headline neither.

**Q: A fixed best camera beats your selection (90.2 vs 98.3 mm). Doesn't that
defeat the method?**
A: For selection, yes — that comparison is part of why we withdrew the
selection claim. Fusion is unaffected: it needs no ranking, only averaging,
and it beats the deployable baseline in both regimes (+23.7% static, +10.6%
dynamic).

**Q: Why is fusion better than selection at 2 views but worse at 8?**
A: With few views, averaging suppresses independent errors and no single view
is clearly best. With many views, the pool contains genuinely bad views whose
inclusion drags a mean-based fusion down, while selection can ignore them —
median fusion sits between the two because it is robust to that minority.
Practical rule: fuse when you have 2-3 views, select when you have 5+.

**Q: Is this just triangulation?**
A: No. Triangulation requires camera extrinsics and 2D correspondences. Here
each view is independently lifted to 3D and canonicalized into a body-fixed
frame; fusion is then a weighted average in that shared frame. No calibration,
no correspondence search, no training — which is also why it does not reach
true triangulation accuracy.

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
| Multi-view selection & fusion | `thesis_artifacts/fusion/fusion_results.json` |
| H36M baseline reproduction (45.1mm) | `thesis_artifacts/baseline_results.json` |
