# Result: the fifteenth and sixteenth. Neither beats Kabsch.

Run 2026-08-07, after `PREREGISTRATION.md` was committed (7e09462).
Artifacts: `bestframe/bestframe{,_motionbert}.json`,
`asymmetric/asymmetric{,_motionbert}.json`.

---

## Experiment 15 — Reading 2. The comparison was not decided by frame choice.

Scored on the nine joints that are a constructor for no variant.

| variant | XS mm | gain vs template | MB mm | gain vs template |
|---|---|---|---|---|
| `both` (published default) | 115.01 | −49.17 | 91.74 | −27.32 |
| `hip_only` | 114.46 | −48.62 | 91.01 | −26.59 |
| `shoulder_only` | 108.16 | −42.32 | **86.87** | −22.45 |
| `weighted` | 109.67 | −43.83 | 93.43 | −29.01 |
| **`svd`** | **107.33** | **−41.49** | 101.94 | −37.52 |
| template Kabsch | **65.84** | — | **64.42** | — |

All-seventeen joints, where the published figures live:

| variant | XS mm | MB mm |
|---|---|---|
| `both` (published) | 75.32 | 60.09 |
| `shoulder_only` | **71.37** | **57.45** |
| template Kabsch | **53.03** | **51.27** |

**Verdict: no construction beats Kabsch, on either backbone, at either scoring
set. Every one of the five is `template_better` with the interval excluding
zero.**

**What it does establish, and it is worth having.** The published comparison used
`both`, and this report's own confirmed level-one result says that is not the
best construction. Running the best one closes the question: the best variant
is **7.68 mm better than the default on XS** and 4.87 mm on MB at the nine-joint
scoring set (2.64 mm over all seventeen), and the gap to
Kabsch narrows from 49.17 to 41.49 mm (XS, nine joints) and from 22.28 to 18.33
(XS, all seventeen). **The conclusion survives the correction, with a smaller
margin than the report currently states.**

An examiner asking "did you compare your best frame, or your default one?" now
has an answer, and it is the answer that costs us least: we compared the
default, we have since compared all five, and the baseline still wins.

**One honest wrinkle.** The best variant differs by backbone and by scoring set
— `svd` on XS at nine joints, `shoulder_only` everywhere else, and `svd` is
*worse* than the default on MotionBERT (68.93 vs 60.09). There is no single best
construction. That is consistent with the level-one finding, which ranks
constructions by axis length rather than crowning one.

---

## Experiment 16 — Reading 2, and the mechanism behind it is wrong.

Left-side distal joints only `{5, 6, 12, 13}`, against the tenth experiment's
bilateral `{2, 3, 5, 6, 12, 13, 15, 16}`. Same scored set, same threshold.

| σ mm | XS anat | XS t17 | XS verdict | MB anat | MB t17 | MB verdict |
|---|---|---|---|---|---|---|
| 0 | 53.45 | 43.30 | template better | 44.13 | 40.96 | template better |
| 40 | 53.45 | 45.72 | template better | 44.13 | 43.50 | not established |
| 80 | 53.45 | 51.71 | not established | 44.13 | 49.74 | **anatomical better** |
| 160 | 53.45 | 70.68 | anatomical better | 44.13 | 69.43 | anatomical better |

**Crossover at ≤ 80 mm on MotionBERT only. MotionAGFormer-XS not until 160 mm.
One backbone is not two — the same rule that failed the tenth pre-registration,
the conditioning index, and now this. Reading 2, failed.**

### The asymmetry hypothesis is refuted, not merely unproven

The prediction was that one-sided corruption rotates the Kabsch fit *more* than
bilateral corruption, so the crossover should arrive *earlier*. It arrives
later or at the same severity:

| | symmetric (tenth) | asymmetric (this) |
|---|---|---|
| XS, first crossover | 160 mm | 160 mm |
| MB, first crossover | **40 mm** | **80 mm** |
| XS t17 at σ=160 | 92.29 mm | 70.68 mm |
| MB t17 at σ=160 | 91.39 mm | 69.43 mm |

Kabsch is damaged *less* by the one-sided corruption, on both backbones, at
every severity. **The dominant variable is how many joints are corrupted — four
here against eight there — not whether the corruption is one-sided.**

**This comparison is confounded and the confound is fatal to the inference.**
Halving the corrupted set halves the displaced mass, so laterality and joint
count move together and nothing here isolates asymmetry. A clean test compares four left-side joints against four joints split
two-and-two. **That test was subsequently pre-registered and run as the
seventeenth experiment** (`thesis_artifacts/laterality/`), and it refutes the
hypothesis outright: at matched joint count, one-sided corruption damages the
template alignment *less*, by 0.52 mm at σ=80 and 1.53 mm at σ=160, on both
backbones. What can be said from this experiment alone is narrower: corrupting
one side rather than both produced no earlier crossover.

That hypothesis was mine, generated from the mechanism the tenth and eleventh
experiments implied, and a 104-agent literature sweep had already found no
documented regime of the kind. It is recorded as the sixth failed search in this
family, and the seventeenth experiment closed it.

---

## Where this leaves the thesis

Six pre-registered searches for a regime where the anatomical frame beats
Kabsch-to-template have now failed: distal corruption, template mismatch, the
literature, anchor corruption (which found the frame *worse*), best-frame
construction, and asymmetric corruption.

**The report's sentence "we found no pose regime in which our construction is
preferable" is now backed by six searches rather than none, and one of them
tested every frame construction the report knows how to build.** That is a
substantially stronger statement of the limitation than the report currently
makes, and it is the honest direction for it to move.

Nothing here changes an existing number. Seventeen pre-registered experiments; the tally of
failures grows by two.
