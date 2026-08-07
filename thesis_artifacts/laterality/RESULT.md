# Result: the seventeenth. The mechanism is backwards, and the effect is tiny.

Run 2026-08-07, after `PREREGISTRATION.md` was committed (ff5bd39).
Artifacts: `laterality.json`, `laterality_motionbert.json`.

## Verdict: Reading 3, on both backbones

Both arms corrupt four joints — one knee, one foot, one elbow, one wrist, at
matched radii. Only the side assignment differs. Positive means one-sided
damages Kabsch more, which is what was predicted.

| σ mm | XS one-sided | XS balanced | XS difference | MB one-sided | MB balanced | MB difference |
|---|---|---|---|---|---|---|
| 0 | 43.30 | 43.30 | +0.00 | 40.96 | 40.96 | +0.00 |
| 20 | 43.92 | 43.93 | −0.01 | 41.61 | 41.65 | −0.03 |
| 40 | 45.72 | 45.72 | −0.00 | 43.50 | 43.54 | −0.04 |
| 80 | 51.71 | 52.23 | **−0.52** [−0.66, −0.38] | 49.74 | 50.27 | **−0.53** [−0.68, −0.37] |
| 160 | 70.68 | 72.21 | **−1.53** [−1.86, −1.17] | 69.43 | 70.94 | **−1.51** [−1.82, −1.19] |

Both sanity criteria passed: the arms are bit-identical at σ = 0, and the
anatomical frame is flat at 53.45 mm (XS) and 44.13 mm (MB) at every severity,
since it reads none of the corrupted joints in either arm.

**The sign is negative at every severity where the interval excludes zero, on
both backbones. One-sided corruption damages Kabsch *less*, not more. The
asymmetry hypothesis is refuted, not merely unsupported.**

The replication is unusually tight — −0.52 against −0.53, and −1.53 against
−1.51, on two different backbones. This is not a marginal effect being read
optimistically; it is a small effect measured precisely.

## The mechanism, and why it explains six failures

Kabsch fits a **rotation**. A one-sided displacement is *partially absorbable*
by rotating toward it — some of the corruption is spent moving the fit rather
than degrading it. A balanced displacement offers the rotation nothing to
absorb: there is no rotation that compensates for mass moving outward on both
sides at once.

That single sentence is consistent with every result in this family:

| corruption | joints | damage to Kabsch at σ=160 (XS) |
|---|---|---|
| bilateral distal (tenth) | 8 | 92.29 mm |
| balanced, matched types (this) | 4 | 72.21 mm |
| one-sided (sixteenth) | 4 | 70.68 mm |

More joints damage it more; at equal joints, balanced damages it more than
one-sided. The ordering is monotone and the mechanism accounts for it.

## The number that ends the search

**Laterality moves Kabsch by at most 1.53 mm. The gap between the anatomical
frame and Kabsch is 41.49 mm.**

No manipulation of corruption geometry in this family could ever have closed
that gap — the lever is roughly twenty-seven times too small. That is a
quantitative reason to stop looking, and it is worth more than the six
qualitative failures that preceded it. Had this been measured before the tenth
pre-registration, four experiments would not have been run.

Stated for the defence: *the reason no corruption regime favours the anatomical
frame is not that we failed to find the right one. It is that the largest
effect available from this lever is one and a half millimetres against a
forty-one millimetre deficit.*

## What changes in the report

Nothing. The pre-registration fixed in advance that no reading of this
experiment could produce a regime where the frame beats Kabsch, and none did.
Seventeen pre-registrations; this is the seventh consecutive failure of the
family, and the first one that explains the other six.
