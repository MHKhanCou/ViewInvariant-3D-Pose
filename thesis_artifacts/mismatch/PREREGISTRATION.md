# Pre-registration: does the baseline's win depend on the template matching the subject?

Written and committed **before** the experiment is run. Eleventh
pre-registration in this project.

Date: 2026-08-07

## Why this experiment exists

Section 5.10 reports that Kabsch alignment onto a fixed reference skeleton beats
the anatomical frame on all 180 held-out pairs, both backbones, all fifteen
actions, under three unrelated templates and every centring. The tenth
pre-registration looked for a robustness regime under distal joint corruption
and **failed** its own criterion — the crossover appeared on one backbone, not
two.

This tests the baseline's other structural requirement, and the one it cannot
remove. **Kabsch-to-template needs a reference skeleton whose proportions
resemble the subject's.** The anatomical frame does not: it reads a hip axis and
a torso axis off the subject's own body, so it has no reference skeleton to
mismatch. If a clinic applies a template built from adult data to a child, or to
anyone whose limb-to-torso ratio is unusual, the baseline's requirement profile
is strictly larger than ours in a way none of the three template ablations
tested — all three used adult templates of ordinary proportion.

**This is again a regime chosen because we expect to win in it.** Saying so is a
condition of running it. Two things keep it honest: the pre-registered
crossover threshold below, and the fact that a null result here *strengthens*
the baseline and will be reported as such.

## Design

Human3.6M, the same 180 held-out camera pairs, same 5 Hz subsample, same
validity mask, same cluster bootstrap over the thirty subject-action groups,
both backbones. Scored on the **thirteen non-constructor joints**, matching the
published headline exactly, so every number is directly comparable to
Section 5.10.

Only one thing varies: the template's body proportions.

**Retargeting.** The template is the mean canonical MPI-INF-3DHP pose, as in
`evaluation/template_baseline.py`. Its limb bones — the eight leading to knees,
feet, elbows and wrists — are scaled by a factor *f*, walking the H36M kinematic
tree from the root so downstream joints follow their parents. Torso, hip,
shoulder, neck and head bones are left untouched, so *f* changes the
limb-to-torso ratio and nothing else.

`f ∈ {0.6, 0.8, 1.0, 1.2, 1.4}`. **f = 1.0 is the identity control** and must
reproduce the published result. f = 0.6 is roughly child-like proportion;
f = 1.4 is unusually long-limbed.

**Why non-uniform.** Kabsch here is rotation-only, and a uniformly scaled
template yields the identical rotation, because scaling the cross-covariance
matrix by a positive constant leaves its SVD rotation factors unchanged. A
uniform scale is therefore a no-op and is asserted as such in the code. Only a
change in *proportion* can affect the fitted rotation.

**Three arms**, as in the tenth pre-registration:

| Arm | Alignment | Reference it needs |
|---|---|---|
| **A** anatomical | Gram-Schmidt body frame (ours) | none |
| **B** template-17 | Kabsch onto the retargeted template, all 17 joints | the template |
| **C** template-4 *(control)* | Kabsch onto the same template, constructor joints only | the template |

## Criteria, fixed before the numbers exist

Primary comparison is the paired per-pair difference A − B at each *f*, with a
cluster bootstrap CI over the thirty groups.

1. **Sanity.** At f = 1.0 the run must reproduce the stored Section 5.10 figures
   to within 0.05 mm — anatomical 93.35 mm and template 57.47 mm on
   MotionAGFormer-XS. If it does not, the retargeting code has altered the
   identity case and the sweep is void.
2. **Crossover.** The baseline's advantage is conditional on template match only
   if A is lower than B, CI excluding zero, at some `|f − 1| ≤ 0.4` **on both
   backbones**. One backbone is not two — the rule that rejected the
   conditioning index and the tenth pre-registration.
3. **Control.** If criterion 2 fires, compare A against C at the same *f*. If C
   is within 10 % of A, the effect is the joint subset rather than the absence
   of a reference skeleton.

## The three readings, fixed before the number exists

1. **Crossover on both backbones within |f − 1| ≤ 0.4, and C does not match A.**
   The baseline's dominance is conditional on a template that matches the
   subject's proportions. Reported with the tolerance stated — never as
   overturning Section 5.10, which used a matched template and stands.
2. **Crossover on one backbone only, or only beyond |f − 1| = 0.4.** Does not
   replicate. Reported as the eleventh pre-registration, failed.
3. **No crossover at any f tested.** The baseline is robust to body-proportion
   mismatch as well, and its advantage is unqualified on this data. **This is
   the outcome that strengthens the competitor, and it will be reported in those
   words**, in Section 5.10 and in Limitations.

## What gets written, whichever fires

One paragraph in Section 5.10 and one table. **No inference beyond the number** —
nothing about the contribution list, the title, or the framing. Readings 2 and 3
change no existing sentence in the report beyond adding the outcome.

## Isolation

New module `evaluation/template_mismatch.py`, new artifact directory
`thesis_artifacts/mismatch/`. No file producing an audited number is modified.
All three arms are derived in a single run so the comparison is paired.

## Cutoff

If this is not complete by 8 Aug 2026, 18:00, it is abandoned and does not enter
the report. The defence is 9 Aug.
