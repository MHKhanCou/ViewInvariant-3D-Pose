# Pre-registration: the seventeenth experiment — laterality, with the confound removed

Written and committed **before** the run. Date: 2026-08-07.

## Why this exists, and why it is not fishing

The sixteenth experiment compared **four left-side distal joints** against the
tenth's **eight bilateral distal joints** and found the one-sided corruption
damaged Kabsch *less*, not more. Its result record states plainly that the
comparison is confounded: laterality and joint count moved together, so nothing
in it isolates asymmetry.

That confound was identified and written down in
`thesis_artifacts/bestframe/RESULT.md` **before this experiment was designed**,
and the fix is mechanical rather than exploratory — hold the number of
corrupted joints and their anatomical types fixed, and vary only which side
they sit on. No new arm is being added in search of a favourable number; one
arm is being added to make an existing comparison interpretable.

This is the same move the report already makes with the circularity control
that demoted the multi-scale variant: a control that can only clarify, never
flatter.

**Six pre-registered searches for a regime where the anatomical frame beats
Kabsch have failed. This is not expected to be the seventh success, and it is
not designed to be — it is designed to explain the sixth failure.**

*Updated 8 Aug 2026: "six pre-registered" is loose here, in the same way it is in
`../bestframe/RESULT.md`, where the tally is set out and corrected. The report no
longer quotes a count for that reason. This pre-registration is left unedited as
the record of what was committed before the run.*

## Design

Identical to `evaluation/asymmetric_corruption.py` in every respect — same 180
held-out Human3.6M pairs, same 5 Hz subsample, same scored set `{9, 10, 11, 14}`
(clean and non-constructor), same severities, same cluster bootstrap, both
backbones — except that two corrupted sets are run side by side:

| arm | joints | knee | foot | elbow | wrist | sides |
|---|---|---|---|---|---|---|
| **one-sided** | `{5, 6, 12, 13}` | l | l | l | l | 4 left, 0 right |
| **balanced** | `{5, 3, 12, 16}` | l | r | l | r | 2 left, 2 right |

Both arms corrupt **four** joints, **one of each anatomical type** (knee, foot,
elbow, wrist), at the same radii. The only difference is the side assignment.
This is the comparison the sixteenth experiment could not make.

The anatomical frame reads `{0, 1, 4, 8}` and is untouched by either arm, so it
is flat by construction in both and serves as the internal control.

## Criteria, fixed before the numbers exist

Primary quantity: the template-Kabsch cross-view distance under each arm, and
the paired per-pair difference **one-sided minus balanced**, cluster-bootstrapped
over the thirty subject-action groups.

1. **Sanity.** At σ = 0 the two arms must be identical to within 0.01 mm, since
   neither corrupts anything. If they differ, the harness is wrong.
2. **Sanity.** The anatomical arm must be flat across all severities in both
   arms, since it reads none of the corrupted joints.
3. **Primary.** One-sided corruption damages Kabsch **more** than balanced
   corruption — the difference positive with CI excluding zero — at some σ, **on
   both backbones**.

## Readings, fixed before the numbers exist

1. **One-sided damages Kabsch more, both backbones.** Laterality is a real
   factor and the sixteenth experiment's null was a joint-count artifact. The
   mechanism stated in the tenth and eleventh result records is confirmed, and
   the honest next step — real unilateral occlusion rather than injected noise —
   is named as future work, not claimed.
2. **No difference, or a difference on one backbone only.** Laterality does not
   matter at matched joint count. The dominant variable is how many joints are
   corrupted, exactly as the sixteenth's confounded comparison suggested, and
   the asymmetry hypothesis is closed rather than merely unsupported.
3. **Balanced damages Kabsch more.** The mechanism is backwards and both the
   tenth and sixteenth result records must say so.

**None of the three readings produces a regime where the anatomical frame beats
Kabsch.** This experiment cannot deliver that and is not being run for it. It
resolves why the previous one failed, which is worth one evening and nothing
in the report has to change for any outcome.

## Isolation and cutoff

New module `evaluation/laterality_control.py`, new artifact directory
`thesis_artifacts/laterality/`. No file producing an audited number is modified.
Abandoned if not complete by 8 Aug 2026, 20:00.
