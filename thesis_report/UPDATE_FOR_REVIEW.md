# Update for review — SUPERSEDED SNAPSHOT (6 Aug 2026)

> **Stale.** The frozen report is 90 pages, 255 audit claims, 76 tests, 28
> references. Numbers below reflect an earlier state; trust DEFENSE_QA.md and
> the report itself.

Thesis: training-free body-frame canonicalization for cross-view comparability of
frozen monocular 3D pose predictions. Report due 6 Aug, defense 9 Aug. This
supersedes all earlier versions of this file.

**Current state: 88 pages, 180/180 artifact-consistency claims, 72/72 tests,
abstract on one page, all 24 references cited.**

The research has been frozen for several days. Everything since has been
correctness work, and it has cost the thesis **five** claims. Each was withdrawn
because a check that should have been run earlier was finally run. Listed newest
first, because the newest is the largest.

---

## 1. The multi-scale result is largely mechanical, and is demoted

**This is the most consequential change and it came from an external review.**

The per-limb frames are built from exactly the joints they are then scored on. In
the long-axis definitions:

```
left_arm   ids [14,15,16]   y=(14,15)  x=(15,16)
right_arm  ids [11,12,13]   y=(11,12)  x=(12,13)
left_leg   ids [1,2,3]      y=(1,2)    x=(2,3)
right_leg  ids [4,5,6]      y=(4,5)    x=(5,6)
```

Root-centred, a three-joint segment has two free points and so six coordinates. A
Gram-Schmidt frame built from those same three points removes three rotational
degrees of freedom, leaving the configuration determined by two lengths and one
angle. **Such a measurement cannot show orientation disagreement** — orientation
was removed by construction, not by the method working.

`evaluation/multiscale_control.py` compares every level against a Procrustes
oracle on the *same* joints, which is the floor no rotation can beat:

| Level | Builders | Canonical | Headroom over floor |
|---|---|---|---|
| **Global frame** | 4 of 17 | 76.2 mm | **1.46×** |
| Long-axis right arm | **3 of 3** | 27.3 mm | **1.13×** |
| Long-axis left arm | **3 of 3** | 26.7 mm | **1.15×** |
| Long-axis legs | **3 of 3** | 30.1 / 31.6 mm | **1.23×** |

**The comparison that settles it:** moving the right arm from the shipped
definition to the long-axis definition takes canonical distance from **58.0 to
27.3 mm while the oracle stays at 24.3 mm**, because the joint set never changed.
The entire gain coincides with adding a third constructor joint.

The rise from 25.6 to 55.1 percent is therefore **demoted from a result to an
exploratory measurement**, in the abstract, the contributions, Chapter 5 and the
README.

### It also broke the evidence for the axis-length principle

| Correlation over the 8 limb levels | |
|---|---|
| ρ(axis length ratio, distance) | **+0.904** ← the previously reported figure |
| ρ(constructor count, distance) | **−0.850** |
| ρ(axis length ratio, constructor count) | **−0.850** ← not separable |

The short-axis levels are precisely the two-constructor levels. With eight points
where both vary together by construction, no analysis of this set can attribute
the effect to either, and normalising by the oracle does not help because the
confound is between the predictors.

**What survives is one clean test.** §5.16 compares two *global* frames, hip axis
versus the longer shoulder axis. Both are scored on all seventeen joints and both
take four constructor joints, so joint set, scored set and constructor count are
identical and only axis length differs. The shoulder wins on both backbones, 5.2%
and 4.4%, intervals [+1.6, +7.8] and [+1.7, +3.6] excluding zero. **That is now
the stated evidence for the principle**; the limb-level correlation is demoted to
illustration.

**The headline 74.1% cross-view result is unaffected.** Its frame takes 4
constructor joints of 17 scored, thirteen held out, and sits 46% above its own
Procrustes floor with real orientation disagreement still measurable. The
circularity is a property of three-joint limb segments, not of the method.

## 2. The articulation boundary is validation, not discovery

A reviewer asked the deciding question: has biomechanics or motion capture
already observed that rigid-body propagation breaks at articulated joints? It
has, from two directions — kinematic-chain error accumulation is standard (1° at
the shoulder becomes centimetres at the wrist; the ankle as chain endpoint
carries the most), and wrists and ankles top the per-joint error table in every
published Human3.6M paper.

Three places implied "neither literature had reason to test this". All rewritten;
the item is retagged **[Negative result] → [Validation]**. What it still
establishes is narrower: the *radius-only model* of per-joint disagreement, which
the two preceding levels make natural to reach for, is the wrong model — tested
pre-registered, parameter-free, with a matched-radius control.

## 3. The propagation is prior art, not our derivation

Verified after a reviewer flagged that this sentence had become the heart of the
thesis. Della Croce, Cappozzo & Kerrigan (1999) and Della Croce, Leardini, Chiari
& Cappozzo (2005) propagate anatomical landmark error into frame orientation and
joint kinematics, and carry the pivot relation `e = l·tan α` — which *is* the
Level-3 prediction we test and find fails. Equation (3.2) is a restatement.

Separately, the construction is **TRIAD** (Black, 1964) and the primary-axis rule
is **Shuster & Oh (1981)**. The same literature explains a failure the report
previously could not: the SVD variant is unweighted **Wahba**, optimal only under
inverse-variance weighting and independence, both of which fail here.

## 4. A claim withdrawn for having no confidence interval

c = 21.3 [19.1, 42.4] and 21.5 [19.2, 42.7]. Interval wider than the estimate, so
"the constants agree to within one percent" carried no weight. Withdrawn in four
places. Worse: those numbers came from a scratch script never committed, so
nothing verified them.

## 5. The headline percentage did not reproduce from its own table

"320.4 mm to 75.3 mm, an improvement of 74.1 percent" — that arithmetic gives
76.5. All improvement figures are per-pair means, not ratios of aggregate means.
The per-pair convention is the conservative one in every case; it is now
disclosed where a reader first reaches for a calculator, with audit claims
pinning both.

---

## What the thesis claims now

| Level | Question | Verdict |
|---|---|---|
| Between frame **constructions** | What should the frame be built from? | **Holds** — on the clean global-frame test only |
| Between **frames** of one construction | Which frame to trust? | **Fails** — indistinguishable from a random null |
| Between **joints** of one frame | Which joint disagrees? | **Fails** — articulation dominates (phenomenon known; the model-rejection is ours) |

Supporting: 74.1% cross-view **agreement** improvement over 180 held-out H36M
pairs (179 improve, 90.5% of the oracle gap), replicating at 77.5% on a second
lifting backbone with unmodified code; the falsified reliability score gating
canonicalization quality on both backbones (exploratory); the bone-length
retraction.

## Terminology now used consistently

- **"cross-view agreement"**, never "improved 3D estimation" — the frozen
  estimator's output is unchanged; only its coordinate frame changes.
- **"artifact-consistency audit"** — it verifies stored JSON against reported
  numbers; it does not re-run experiments.
- **"two distinct lifting backbones"**, not "independent" — both are temporal
  transformers on the same benchmark family.
- **"version-controlled prospective analysis plan"** where precision matters,
  though the report still says pre-registered and describes exactly what that
  means.

## Corrections outside the report

- The README cited three papers wrongly (VideoPose3D under the wrong title and
  author, MotionBERT under the wrong year, P-STMO under a wrong name) and
  contradicted the thesis bibliography. Fixed.
- A test imported Ultralytics at module scope and failed on a clean machine, in a
  file whose docstring promises it runs anywhere. Now lazy with a skip.
- Counts reconciled to one authoritative pair: **180 claims, 72 tests**.
- BibTeX `@mastersthesis` → undergraduate thesis.
- **The title was never supervisor-approved.** Two documents said or implied it
  was; both now ask him to confirm. A different, overclaiming title string had
  also propagated into the README BibTeX and was corrected against the report.

## Where the work is weakest — unchanged

1. **Agreement is not correctness.** Two predictions wrong the same way agree
   perfectly. Bounded by the oracle floor and the retained correlation with
   ground-truth error, but that check exists on one dataset.
2. **No downstream task succeeds.** The honest ceiling on significance.
3. **Two datasets, both lab rigs; two backbones, both transformers.**
4. **The triage finding is exploratory** — a comparator in its pre-registration.
5. **Confidence intervals are conditional** on the selected H36M subjects,
   actions and cameras; there are only two test subjects.

## Questions worth putting to a reviewer

1. After the multi-scale demotion, is the remaining evidence for the axis-length
   principle — one clean global-frame comparison on two backbones — enough to
   carry it as a contribution, or should it too drop to exploratory?
2. Five withdrawals in one week, all self-found and each with an audit claim
   behind it. Does that read as rigour, or does it invite doubt about what else
   is unchecked?
3. Is anything still overstated?
