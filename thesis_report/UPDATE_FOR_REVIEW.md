# Update since the last plan

Thesis: view-invariant 3D human pose estimation. Report due 6 Aug, defense 9 Aug.
This covers everything done since the one-day sprint plan was written.

Current state: **74 pages, 150/150 audit claims verified against stored
artifacts, 72/72 tests passing.** Every number below traces to a JSON artifact
via `python -m evaluation.audit_numbers`.

---

## 1. The conditioning-abstention experiment (plan item 1, primary)

The plan was to build a label-free successor to the analytic reliability score
out of the geometry the report derives in Section 3.2.

**Before implementing, the implication was checked mathematically, and it did not
hold in the form first written down.** `theta_rms ~= 2*sigma/L` describes the
distribution of angular error for a given axis length, not one frame's
realisation of it. Measured over 52,900 frames, the lateral axis varies by
CV 5.5 percent and p99/p1 = 1.29 *within* a construction, against 3.34 *between*
constructions - roughly 4.6x less variation in the regime the experiment would
have run in. The index is also computed from the prediction, so a distorted pose
raises both the index and the error; a positive correlation would have been
uninterpretable, which is exactly the trap the bone-length signal fell into.

So the pre-registration was rewritten to the narrower question the theory does
support - can conditioning identify a tail to abstain on - with a coverage-error
curve against a random-ordering null and a partial correlation given bone-ratio
deviation as the confound control. Committed as `fc2428a` before any number
existed.

**Result: the pre-registered criterion FAILED.** Conditioning passes on
MotionBERT and fails on MotionAGFormer. The pre-registration required both, so it
is reported as not replicating.

**The comparator is the finding.** The analytic reliability score - the component
the report spends five falsifications demolishing - passes the same test on both
backbones:

| | MotionAGFormer-XS | MotionBERT |
|---|---|---|
| mean canonical distance, 0% abstained | 76.2 mm | 60.8 mm |
| 30% abstained | **59.5 mm** | **54.7 mm** |
| gain over random @10% | +5.44 mm, 95% CI [+1.53, +9.27] | +2.74 mm [+1.55, +3.81] |
| worst 5% tail ratio | 1.54x | 1.49x |
| partial rho given bone deviation | -0.306 | -0.322 |

Both intervals exclude zero. Both survive the confound control.

**This retracts nothing.** The five falsifications tested the score against
*ground-truth pose error*, and it still does not predict that. They tested the
wrong target for part of its purpose: `reliability.py`'s own docstring says the
score estimates whether the body-frame axes are reliable enough to canonicalize
with, and against *that* target it works on two backbones.

Consequence for the thesis: "Reliability-Aware" in the approved title is now
defensible, for a narrow claim. Stated as **exploratory** everywhere it appears,
because it was a comparator in that pre-registration, not its subject.

Commit `63ddd6e`.

---

## 2. The joint-level test of the axis-length law (unplanned, added after)

The law had been tested *between frame constructions* (confirmed) and *between
frames of one construction* (failed, above). The level the derivation speaks to
most directly had never been tested: **across the joints of one canonicalized
pose.**

A frame carrying rotation error `Theta` displaces a joint at `p_j` by
`Theta x p_j`, so the frame-induced part of cross-view distance should be linear
in that joint's radius. A per-frame Procrustes rotation separates that part from
the estimator's own error.

**What made it worth doing: nothing is fitted.** `sigma = 7.5 mm` came from the
limb-frame fit, `L_hip = 275.8 mm` and `L_torso = 456.1 mm` from independent
measurements, giving a predicted slope band of **[0.038, 0.073]** before any data
was read. The Procrustes bias was recorded in advance as upward, so a slope
*below* the band would count for more than one above. Pre-registered in
`be19314`.

**All three predictions failed.** Slope 0.218 and 0.105 against the band;
R² 0.339 and 0.337 against a 0.80 threshold. On MotionBERT the confound control
failed too (1.30x vs required 2.0), which the pre-registration had already named
as voiding the decomposition there. Dropping the frame-defining joints made it
worse (R² = -0.070).

`tests/test_radial_law.py` includes a **positive control**: on synthetic views
differing by a pure rotation, the same code path recovers R² > 0.80. So the
hypothesis failed, not the arithmetic.

**Why it failed (post-hoc, labelled as such, replicates on both backbones):**

| Joint group | mean radius | cross-view distance |
|---|---|---|
| Rigid with torso (neck, head, shoulders) | 566 mm | **71.9 mm** |
| Beyond a hinge (knees, feet, elbows, wrists) | 532 mm | 197.5 mm |

Larger radius, *less* disagreement - the radius model is contradicted in the
wrong direction to be a matter of degree. Matched pairs remove any grouping
choice: left shoulder vs left knee differ by **1.2% in radius and 2.62x in
distance** (MotionBERT: 0.9% and 2.06x). The governing variable is position in
the kinematic chain relative to the frame's defining segment, not Euclidean
radius.

**A flattering artifact this caught in our own evaluation:** canonicalization
looks near-perfect at the thorax (22.1 mm) and hips (54.4 mm) *because the frame
is built from those joints and the construction pins them*. Any per-joint average
over all seventeen joints silently absorbs this and overstates the method. Our
per-joint figures now separate them.

Commit `7794907`.

---

## 3. The resulting three-level bound

Three pre-registered tests, two of them failures, now delimit the report's
central design principle:

| Level | Question | Verdict |
|---|---|---|
| Between frame **constructions** | Which frame to build? | **Confirmed** - rho +0.904 / +0.880, implied sigma agrees to 1% across two backbones; acting on it raises multi-scale improvement 25.6% -> 55.1% |
| Between **frames** of one construction | Which frame to trust? | **Fails** - within a construction the axis is an anatomical near-constant |
| Between **joints** of one frame | Which joint will disagree? | **Fails** - articulation dominates geometry |

This is now presented as the primary contribution rather than as three scattered
results.

---

## 4. Narrative reframe (commit `492e3b7`)

Canonicalization of 3D pose is occupied prior work - 3DPCNet (ICASSP 2026) does
it with a learned network, and the report already cites and tabulates it. Leading
with "a training-free canonicalization framework" therefore puts the weakest
claim first.

The question the report actually answers is not occupied: *what determines
whether an anatomical reference frame is consistent across viewpoints.* It was
buried - derived in Section 3.2, tested across Chapter 5, third in the
contributions list, absent from the abstract's opening.

Changed: the abstract now states the question, the derivation and the three-level
result **and still fits on one page**; the contributions list leads with the
principle and its delimitation, framework demoted to the vehicle; the conclusion
carries the full arc. Narrative only - nothing re-run, no artifact touched.

Two stale figures fixed: the abstract claimed the reliability score was falsified
along *seven* axes when Section 5.9 reports five, and `DEFENSE_QA.md` claimed
97 tests when `unittest` discovers 67 (now 72).

---

## 5. Report corrections found while writing the above (commit `599008b`)

- **Future Work asked for a second frozen backbone** as "the single most valuable
  outstanding experiment". That experiment is in Section 5.15. Replaced with the
  gap that actually remains: both backbones are transformers.
- **Section 5.17 contradicted Section 4.3.** The cross-view retrieval result
  predates the protocol correction, uses fifty near-identical standing poses, and
  starts from Recall@1 = 0.02 - close to chance before canonicalization touches
  it. Now reported as a bound, not a measurement, with the proper experiment
  moved to Future Work.
- **CHAMP and CUPS added as [17] and [18].** The report twice said it offers no
  coverage guarantee "of the kind conformal methods provide" without citing any.

---

## 6. Plan status, including one item that died

| Plan item | Status |
|---|---|
| 1. Conditioning abstention (primary) | Done - failed as pre-registered, produced the triage finding |
| 2. Cross-view retrieval (secondary) | **Cut deliberately.** Needed a gallery protocol, a per-backbone run and a careful non-comparison to Pr-VIPE; a null result on the last day would buy a fourth negative in a report already carrying three. The Section 3.7 / 5.14 contradiction it was meant to fix was fixed directly instead |
| 3. View-count curve ("free", 20 min) | **Dead.** The curve on disk has `weighted_mean` (reliability-weighted), `median`, `reliability_pick`, `mean_single`, `oracle_best` - **no unweighted mean**, which is what the Section 5.11 crossover question is about. Recovering it means re-running `fusion_eval.py`, which rewrites an audited artifact the day before freeze. It stays a stated limitation |

---

## 7. Presentation layer - honest status

**Built and working:**

- Gradio app (`app.py`), Image and Video tabs: 2D keypoint overlay, 3D canonical
  skeleton, avatar render, coordinate-space toggle, rotation control
- `presentation/bvh_export.py` - body-relative BVH, no camera, 7 tests
- `presentation/avatar_renderer.py`
- `canonical/visualization.py`
- 16 figures in the report, all generated from stored artifacts

**Against the original proposal:** the proposal listed four datasets; two were
used, both multi-camera, which is what the central claim requires. CMU Panoptic
and CASIA Gait were not used, and **no gait or sports application was
evaluated.**

**The honest gap, and it is a real one.** The demo shows the *pipeline* - one
video in, canonical skeleton out. It does not show the *result*. The thesis's
central claim is that two cameras viewing the same instant produce poses that
agree after canonicalization, and nothing in the presentation layer demonstrates
that. A two-view side-by-side viewer with the live cross-view distance and the
oracle floor is the one demo that would, estimated at 2.5 h with medium
engineering risk since the app has no notion of two synchronized inputs. It is
the only build task left and it changes no claim.

---

## 8. Where the work is weakest - stated for review

1. **The headline metric measures agreement, not correctness.** Cross-view
   distance falls if both predictions are wrong the same way. Guards: the
   Procrustes oracle floor (canonicalization closes 90.5% of the gap to it), and
   the multi-scale distance retaining its correlation with ground-truth error
   (+0.610 vs +0.601). The second check exists on one dataset only. **This is the
   sharpest available attack.**
2. **No downstream task succeeds.** Retrieval is negative and its protocol is
   superseded. The work improves a metric and never shows the metric buys
   anything. This is the honest ceiling on its significance.
3. **Two datasets, both lab multi-camera rigs; two backbones, both transformers.**
   "Model-independent" rests on n=2.
4. **The triage finding in section 1 above is exploratory**, not the subject of
   its pre-registration. It replicates on two backbones and survives one confound
   control, which is more than the bone-length signal ever had - and that signal
   still died on the second dataset.

## 9. Specific questions

1. Is leading with the axis-length principle rather than the framework the right
   call, given that two of its three levels are failures?
2. Is reporting the reliability-triage result at all defensible given it was a
   comparator rather than the pre-registered subject, or should it be confined to
   a remark?
3. Is the two-view viewer worth 2.5 h on the last day, or is defense rehearsal a
   better use of it?
