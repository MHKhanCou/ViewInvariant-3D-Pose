# Pre-registration: geometric conditioning as an abstention criterion

Written and committed **before** the experiment is run, so version history shows
the criterion preceding the number. Same practice as
`thesis_artifacts/tta/PREREGISTRATION.md` and
`thesis_artifacts/multilandmark/PREREGISTRATION.md`.

Date: 2026-08-05

## Why the obvious version of this experiment was rejected

The first draft of this pre-registration asked whether
`kappa = r_bar / L_hip` predicts a frame's canonical cross-view distance
continuously, with a pass criterion of Spearman rho >= +0.30. That question is
weakly supported by the theory and was discarded before running. Two reasons,
both checked first.

**1. The predictor has almost no dynamic range per frame.** Section 3.2 derives
`theta_rms ~= 2*sigma/L`, which is a statement about the DISTRIBUTION of angular
error for a given L, not about one frame's realisation. It predicts well between
frame constructions because L there spans 138 to 461 mm, a ratio of 3.34. Within
a single construction L is essentially the subject's hip width, an anatomical
constant. Measured over 52900 frames: mean 275.8 mm, CV 5.5 percent, and a
p99/p1 ratio of only 1.29, with a median within-video CV of 4.1 percent. Feeding
that range through the fitted law gives a predicted spread in d of roughly 41 to
53 mm, which is small beside the observed per-frame variance.

**2. The predictor is confounded with the thing it predicts.** kappa is computed
from the prediction. A badly predicted pose has a distorted hip width, which
raises kappa, and is also canonicalized badly, which raises d. A positive
correlation would therefore be ambiguous between "ill-conditioned geometry causes
a bad frame" and "a bad prediction causes both". This is the same confound that
made the bone-length signal uninterpretable until it was tested against a second
dataset.

Forcing a continuous prediction the theory does not support would repeat a
mistake this report already documents.

## The question actually being asked

Can geometric conditioning identify frames on which canonicalization should be
**abstained from**, because the frame itself is ill-posed?

This is the limiting regime the derivation speaks to with confidence: as
`L -> 0`, or as the two axes approach collinearity, the Gram-Schmidt construction
becomes ill-conditioned and the resulting frame is not merely noisy but
meaningless. Triage is a weaker claim than regression, and it is the one an
abstention rule actually needs.

## Conditioning index

Computed from a single predicted pose, no labels and no second view:

    kappa   = r_bar / L_hip                     lever arm over baseline
    ortho   = sin(angle between torso and hip axes)
    cond    = kappa / max(ortho, eps)           combined, higher is worse

`cond` is the criterion; `kappa` and `ortho` are reported separately.

## Primary metric: coverage-error

Sort frames by `cond`, discard the worst fraction, and measure mean canonical
cross-view distance on what remains. Compare three orderings on identical frames:

- `cond` (this criterion)
- the incumbent reliability score (which the report falsifies as a *regressor*;
  this asks the different question of whether it triages)
- random ordering, averaged over 100 shuffles, which is the null

## Predictions

1. **Triage works.** At 90 percent coverage, ordering by `cond` gives a lower
   mean canonical distance than random ordering, with a cluster-bootstrap
   interval over subject-action groups excluding zero.
2. **The tail is real.** The worst 5 percent of frames by `cond` have a mean
   canonical distance at least 25 percent above the pooled mean.
3. **Not merely a distortion proxy.** The advantage over random survives partial
   correlation controlling for bone-ratio deviation
   (`evaluation/bone_consistency.py:66`), which is an independent measure of how
   distorted the predicted skeleton is. This is the confound control; without it
   a positive result is uninterpretable.
4. **Replicates.** 1 and 3 hold on both backbones.

## Pass criterion

Predictions 1, 3 and 4 must all hold. Prediction 2 is reported either way.

Explicitly **not** claimed, and not tested: that `cond` predicts canonicalization
error continuously, or that it can rank frames finely. The theory supports
detecting a tail, so only a tail is claimed.

## If it fails

Reported as a bound, in the same manner as the bone-length retraction and the
multi-landmark refutation: geometric conditioning governs the choice between
frame constructions but does not discriminate between frames of a single
construction, because within one construction the conditioning barely varies. If
so, the report will say that the analytic reliability score's failure is a
special case of a more general limitation, which is a sharper statement than the
one currently given.

## Isolation

No file that produces an audited number is modified. The evaluation is a new
module importing existing helpers and editing none of them.
