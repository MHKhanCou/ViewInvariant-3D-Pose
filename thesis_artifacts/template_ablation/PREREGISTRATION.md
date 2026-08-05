# Pre-registration: two ablations of the template baseline

Written and committed **before** the runs. Seventh pre-registration.

Date: 2026-08-06

Section 5.10.1 reports that Kabsch alignment to a fixed reference skeleton beats
our anatomical frame on all 180 pairs under both backbones. Two questions about
that result were raised immediately and neither is answered by the experiment
that produced it.

## A. Does the win depend on our own method?

The template is the mean **canonical** pose over the MPI-INF-3DHP cache, and
those poses were canonicalized by the construction the baseline is being
compared against. The template's orientation is irrelevant, since it only fixes a
global frame, but its **shape** is a mean over aligned poses and would be blurred
if the poses had not been aligned first. So the baseline may be inheriting
something from the method it beats.

**Test.** Rebuild the template two further ways and re-run unchanged:

1. `raw_first` — the first valid raw MPI prediction, root-centred. One arbitrary
   skeleton, no canonicalization anywhere in its construction.
2. `synthetic` — a neutral standing figure built from median MPI bone lengths
   along fixed anatomical directions. Uses no pose data at all.

**Reading, fixed now.** If the baseline still wins under both, the result is
independent of our method and Section 5.10.1 stands as written. If the win
shrinks materially or reverses under either, the reported comparison depends on
our own canonicalization and Section 5.10.1 must say so.

## B. Does the boundary analysis predict a better template method?

Section 5.19 finds that joints beyond a hinge disagree across views far more than
joints rigid with the torso. A least-squares fit over all seventeen joints is
therefore being dragged by its worst-determined points.

**Test.** Fit the rotation on the nine torso-rigid joints
`{0, 1, 4, 7, 8, 9, 10, 11, 14}` and score on the eight articulated ones
`{2, 3, 5, 6, 12, 13, 15, 16}`. The two sets are disjoint, so nothing is scored
on a joint used to fit — a cleaner separation than anything else in this report.

**Prediction, stated before the run.** Fitting on the rigid subset will beat
fitting on all seventeen, on the same eight scored joints, on both backbones.
This follows from Section 5.19 and is the first place that finding makes a
prediction about a method rather than about an error profile. If it fails, the
boundary analysis has no constructive consequence and we say so.

## What is not being asked

Neither ablation revisits whether the anatomical frame is worth keeping. That
judgement belongs to the evidence as a whole and is not made tonight.

## Isolation

New module reusing `evaluation/template_baseline.py`. No audited file changes.
