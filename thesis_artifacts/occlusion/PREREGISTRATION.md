# Pre-registration: does the anatomical frame have a robustness regime?

Written and committed **before** the experiment is run. Tenth pre-registration in
this project.

Date: 2026-08-06

## Why this experiment exists

Section 5.10 reports that Kabsch alignment onto a fixed template beats the
anatomical frame on all 180 held-out pairs, both backbones, all fifteen actions.
The report's answer to "then why keep the method" is currently *rhetorical*: the
baseline cannot pose the question the thesis asks, because it has no anatomical
axis to hold fixed and vary. That answer is correct and it is not an experiment.

There is one structural difference between the two that is testable. The
anatomical frame reads exactly five joints — {root, r\_hip, l\_hip, spine,
thorax}. Template Kabsch least-squares-fits all seventeen, so error in a wrist or
an ankle rotates the entire alignment. If the distal joints are unreliable, the
baseline should degrade and the anatomical frame should not.

**This is a regime chosen because we expect to win in it. Stating that plainly is
a condition of running it**, and the control arm below is what stops it being
circular.

## Design

Human3.6M, the same 180 held-out camera pairs, same 5 Hz subsample, same validity
mask, same cluster bootstrap over the thirty subject-action groups, both
backbones. Identical to Section 5.10 in every respect except the corruption and
the scored joint set.

**Corrupted joints** (H36M-17 indices) — the eight past a hinge:
`{2 r_knee, 3 r_foot, 5 l_knee, 6 l_foot, 12 l_elbow, 13 l_wrist, 15 r_elbow,
16 r_wrist}`. Isotropic Gaussian noise, added to the cached 3D predictions,
drawn **independently per camera** so it does not cancel in the comparison.
Severities σ ∈ {0, 20, 40, 80, 160} mm. σ = 0 is the identity control.

**Scored joints:** `{9 neck, 10 head, 11 l_shoulder, 14 r_shoulder}`. These are
uncorrupted *and* non-constructor. Both arms are scored on the same four, so the
metric measures damage to the *alignment* and nothing else. It does not measure
the corruption we injected.

**Three arms:**

| Arm | Alignment | Joints it reads |
|---|---|---|
| **A** anatomical | Gram-Schmidt body frame (ours) | the 5 constructors |
| **B** template-17 | Kabsch onto the fixed template | all 17 |
| **C** template-5 *(control)* | Kabsch onto the same template | the 5 constructors only |

Arm C is mandatory. It is arm B told which joints to trust, so it separates *"the
anatomy helps"* from *"reading fewer, more reliable joints helps"*.

Template, corruption seeding and all shared code are reused unchanged from
`evaluation/template_baseline.py`; the RNG is seeded from (pair, camera,
severity) so the sweep is reproducible.

## Criteria, fixed before the numbers exist

Primary comparison is the paired per-pair difference A − B at each severity, with
a cluster bootstrap CI over the thirty groups.

1. **Sanity.** At σ = 0, B must be lower than A on both backbones. If it is not,
   the scored-joint subset has changed the Section 5.10 result and the whole
   sweep is void.
2. **Crossover.** A robustness regime exists only if A is lower than B, with the
   CI excluding zero, at some σ ≤ 80 mm **on both backbones**. A crossover that
   appears only at 160 mm, or only on one backbone, does not count.
3. **Control.** If criterion 2 fires, compare A against C at the same σ. If C is
   within 10 % of A, the effect is attributed to the joint subset, not to anatomy.

## The three readings, fixed before the number exists

1. **Crossover on both backbones and C does not match A.** The anatomical frame
   has a bounded robustness regime the template baseline lacks. Reported as a
   *conditional* advantage with the crossover severity stated, never as
   overturning Section 5.10.
2. **Crossover on both backbones but C matches A.** The advantage is restricting
   the fit to reliable joints, not anatomy. Reported in those words — with the
   honest addendum that the anatomical frame gets that restriction for free and
   arm C had to be told, and with the equally honest note that anyone can build
   arm C in ten lines.
3. **No crossover at σ ≤ 80 on either backbone.** No robustness regime found. The
   template baseline dominates including under distal corruption. Reported as the
   tenth pre-registration, failed, and the rhetorical answer in Section 5.10
   stands alone.

## What gets written, whichever fires

One paragraph in Section 5.10 and one table. **No inference beyond the number** —
nothing about the contribution list, the title, or the framing. Reading 3 changes
no existing sentence in the report.

## Isolation

New module `evaluation/occlusion_robustness.py`, new artifact directory
`thesis_artifacts/occlusion/`. No file producing an audited number is modified.
Arm A is re-derived from the same code path as arms B and C so the comparison is
paired within a single run rather than quoted across runs.

## Cutoff

If this is not complete by 8 Aug 2026, 12:00, it is abandoned and does not enter
the report. The defence is 9 Aug and the report is already complete and
consistent without it.
