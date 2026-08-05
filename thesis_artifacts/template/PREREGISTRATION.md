# Pre-registration: a Procrustes-to-template baseline

Written and committed **before** the experiment is run. Sixth pre-registration
in this project.

Date: 2026-08-06

## Why this experiment exists

Every comparison in this report is against one of two things. Raw camera-frame
predictions, which are not a method but the absence of one. Or the per-frame
Procrustes oracle, which is a floor rather than a competitor, since it needs both
views and no single-view method can reach it.

Neither tells a reader whether the anatomical frame beats the simplest
alternative that satisfies the same requirement profile. That alternative is to
Kabsch-align every pose onto one fixed reference skeleton: training-free,
label-free, calibration-free, single-view, and used by V-VIPE [26] as
preprocessing. **This report has never run it, and a reader is entitled to ask.**

## Where the template comes from, and why not from Human3.6M

The reference skeleton is the mean canonical pose over the **MPI-INF-3DHP**
cached predictions, `thesis_artifacts/cross_view_eval/predictions_cache.npz`.

An earlier draft of tonight's plan said "the development split", which does not
exist here. Section 5.10 states that all 180 pairs and both subjects of
Human3.6M are held out precisely because that dataset played no part in
developing the method, and the cache contains only S9 and S11. There is no
in-dataset, non-test source for a template. Building one from S9 and S11 would
hand the baseline the test distribution while our own method never saw it, which
would make a win meaningless and a loss unfair.

Taking the template cross-dataset is unambiguously disjoint from the evaluation
and, if anything, favours the baseline by giving it a real human skeleton rather
than a synthetic one. That is deliberate: the point of running a competitor is
to run it at its strongest.

## Protocol

Identical to Section 5.10 in every respect except the alignment. Same 180
held-out camera pairs, same 5 Hz subsample, same validity mask, same cluster
bootstrap over the thirty subject-action groups, both backbones. Scored on the
thirteen non-constructor joints, matching the headline established in
`thesis_artifacts/noncon/`, with the seventeen-joint figure reported alongside.

## The three readings, fixed before the number exists

Compare mean cross-view distance under the anatomical frame against the template
baseline, with a cluster bootstrap on the paired per-pair difference.

1. **Template lower on both backbones, interval excluding zero.**
   The anatomical construction is not justified on these data. Section 5.10 says
   exactly that, and Limitations records it as the strongest outstanding
   objection to the method.
2. **Interval spans zero on either backbone.**
   The anatomical frame's advantage over the simplest available alternative is
   **not established**. Reported in those words. It is not reported as a tie in
   our favour, and it is not reported as equivalence.
3. **Anatomical lower on both, interval excluding zero.**
   The comparison supports the construction against the obvious baseline, and it
   is the first such comparison in this report.

## What is written tonight, whichever fires

One paragraph in Section 5.10 stating the outcome, and one line in Limitations.

**No inference beyond the number.** Nothing about what it implies for the
contribution list, the title, or the framing of the thesis. That reasoning
requires a clearer head than the hour this is being run at, and the experiment is
sequenced last precisely so that the report is already complete and consistent
without it.

## Isolation

New module, new artifact directory. No file producing an audited number is
modified, and the anatomical results are re-derived from the same code path so
the comparison is paired rather than quoted across runs.
