# Result: the tenth pre-registration failed

Run 2026-08-06, after `PREREGISTRATION.md` was committed (2014564, corrected in
9c8e787). Artifacts: `occlusion.json` (MotionAGFormer-XS),
`occlusion_motionbert.json` (MotionBERT).

## Verdict: reading 3 — no robustness regime established

**Do not quote the `verdict.reading` field out of `occlusion_motionbert.json`.**
Each run adjudicates its own backbone and cannot see the other. The
pre-registered criterion 2 requires a crossover at σ ≤ 80 mm **on both
backbones**. It fired on one.

| σ (mm) | XS: anat / t17 / t4 | XS verdict | MB: anat / t17 / t4 | MB verdict |
|---|---|---|---|---|
| 0 | 53.45 / 43.30 / 49.69 | template better | 44.13 / 40.96 / 47.05 | template better |
| 20 | 53.45 / 44.59 / 50.32 | template better | 44.13 / 42.32 / 47.71 | not established |
| 40 | 53.45 / 48.05 / 52.05 | not established | 44.13 / 45.98 / 49.50 | **anatomical better** |
| 80 | 53.45 / 59.83 / 58.59 | not established | 44.13 / 58.17 / 56.33 | **anatomical better** |
| 160 | 53.45 / 92.29 / 78.62 | **anatomical better** | 44.13 / 91.39 / 76.80 | **anatomical better** |

MotionBERT crosses over at 40 mm and satisfies the criterion. MotionAGFormer-XS
does not: at 80 mm the difference is +6.38 mm with CI [−2.86, +12.88], which
spans zero, and the interval only excludes zero at 160 mm — a severity the
pre-registration explicitly ruled out in advance.

Criterion 1 (sanity) passed on both: the template baseline is better at σ = 0,
so the scored subset did not change the Section 5.10 result.

## This is the conditioning failure again

The pre-registered conditioning index κ passed on MotionBERT and failed on
MotionAGFormer, and was reported as not replicating. This is the same shape:
**one backbone, not two.** Applying the rule consistently means calling it a
failure, and the rule is only worth anything if it is applied when it costs
something.

## What is true and descriptive, but was not pre-registered

Stated as observation, never as the result:

- The anatomical arm is **exactly flat** across every severity — 53.45 mm on XS
  and 44.13 mm on MB at all five levels — because it reads only {0, 1, 4, 8} and
  none of those was corrupted. That is confirmation of the mechanism, not of the
  claim.
- The sign reverses on both backbones as corruption grows, and by 160 mm the
  margin is large and the interval excludes zero on both (+38.84 and +47.26 mm).
- The control arm C does not rescue the baseline: template Kabsch restricted to
  the same four joints is worse than the anatomical frame at σ = 0 on both
  backbones and still degrades under corruption, because it fits four noisy
  points rather than reading two exact axes.

The honest reading of all three: there is probably a regime, and this experiment
did not establish it at the severity it committed to in advance. Establishing it
would need a corruption model with a defensible severity scale — real occlusion
or real detector failure, not injected Gaussian noise — which is future work and
not two days before a defence.

## Effect on the report

None. The pre-registration fixed this in advance: "Reading 3 changes no existing
sentence in the report." The report's 255 audited claims and its nine
pre-registrations are unchanged; this is the tenth, run after the freeze, and it
failed.
