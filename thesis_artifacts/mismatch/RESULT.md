# Result: the eleventh pre-registration failed, and the competitor got stronger

Run 2026-08-07, after `PREREGISTRATION.md` was committed (1ddcd31).
Artifact: `mismatch.json` (MotionAGFormer-XS), `mismatch_motionbert.json`.

## Verdict: reading 3 — the baseline is robust to body-proportion mismatch

The pre-registration named this outcome in advance and committed to reporting it
in these words: **the baseline's advantage is unqualified on this data.**

| f (limb scale) | anatomical | template-17 | template-4 | t17 − anat | verdict |
|---|---|---|---|---|---|
| 0.60 | 93.35 | 57.59 | 72.54 | −35.76 [−58.88, −19.81] | template better |
| 0.80 | 93.35 | 57.49 | 72.54 | −35.86 [−58.97, −19.89] | template better |
| **1.00** | **93.35** | **57.47** | 72.54 | −35.88 [−58.99, −19.92] | template better |
| 1.20 | 93.35 | 57.49 | 72.54 | −35.86 [−58.96, −19.90] | template better |
| 1.40 | 93.35 | 57.53 | 72.54 | −35.82 [−58.92, −19.85] | template better |

Criterion 1 (sanity) passed exactly: at f = 1.0 the run reproduces the stored
the section "A Single-View Baseline, and It Wins" figures, 93.35 mm and 57.47 mm, to the digit.

Criterion 2 (crossover) never fired. There is nothing to adjudicate under
criterion 3.

## The size of the null is the finding

Scaling the template's limbs to 60 % of their length — roughly child proportion
against an adult template — moves the baseline from **57.47 mm to 57.59 mm.**
That is **0.12 mm, or 0.2 percent.** Lengthening them by 40 % costs 0.06 mm.
The baseline is not merely better; on this axis it is essentially indifferent.

**Why, mechanically.** The baseline fits a *rotation*, and the retargeting is
bilaterally symmetric. Shortening both arms and both legs by the same factor
leaves the point cloud's principal axes where they were, so the optimal rotation
barely moves. A template mismatch would have to be *asymmetric* or *postural* to
rotate the fit, and body-proportion difference between populations is neither.

This was not obvious in advance, and the pre-registration did not predict it. It
is recorded here because the honest reading is that the experiment was aimed at
a weakness the competitor does not have.

## What this closes

Two pre-registered attempts have now looked for a regime where the anatomical
frame beats Kabsch-to-template, and both failed:

| # | Attempt | Outcome |
|---|---|---|
| 10 | Distal joint corruption | Crossover on MotionBERT at 40 mm, MotionAGFormer only at 160 mm — one backbone, not two |
| 11 | Template proportion mismatch | No crossover at any proportion tested; the baseline moves by 0.2 % |

**the section "A Single-View Baseline, and It Wins"'s sentence "We found no pose regime in which our construction is
preferable" now has two further failed searches behind it rather than none.**
That is a stronger statement of the limitation than the report currently makes,
and it is the honest direction for it to move.

The rhetorical defence is unchanged and remains correct: Kabsch has no
anatomical axis, so it cannot host the experiment this thesis runs. But it
should now be stated without any implication that a favourable regime is merely
waiting to be found. Two pre-registered searches looked and did not find one.

## Effect on the report

None required. The pre-registration fixed in advance that readings 2 and 3
change no existing sentence. Both this and the tenth were run after the report
was frozen; the report describes nine pre-registrations and 255 audited claims,
and both remain accurate.
