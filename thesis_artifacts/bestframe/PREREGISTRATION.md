# Pre-registration: the fifteenth and sixteenth experiments

Written and committed **before** either is run. Date: 2026-08-07.

Two last attempts to find a regime where an anatomical frame beats
Kabsch-to-template. Four previous attempts have failed (tenth: distal
corruption, one backbone only; eleventh: template proportion mismatch, 0.2 %;
twelfth: anchor corruption, which found the frame *worse*; a literature sweep
which found the geometric axis occupied). **The prior is poor and is stated
here so that a fifth and sixth failure is not surprising.**

---

## Experiment 15 — was the comparison run against the wrong frame?

### Why this exists

`evaluation/template_baseline.py` calls `canonicalize_stream`, which is the
**default** two-vector construction: hip axis primary. But this report's own
confirmed level-1 result is that **axis length governs the choice between frame
constructions**, and that the longer shoulder axis beats the hip axis — measured
at 5.2 % (XS) and 4.4 % (MB) in `thesis_artifacts/multilandmark/`.

So the headline comparison — the one that concludes a simpler baseline wins —
was run against a construction this thesis's own principle predicts is
suboptimal. That is a genuine gap in the evidence and an examiner is entitled
to it. It is being closed here whichever way it falls.

### Design

Identical to the template baseline in every respect except the frame: same 180
held-out Human3.6M pairs, same 5 Hz subsample, same validity mask, same cluster
bootstrap over thirty subject-action groups, same MPI-derived template, both
backbones. Five constructions from `canonical/multilandmark_frame.py`: `both`
(the published default), `hip_only`, `shoulder_only`, `weighted`, `svd`.

**Scoring set.** The constructions build from different joints, so the
thirteen-joint set is not common to them. Scoring uses the joints that are a
constructor for **no** tested variant — `{2, 3, 5, 6, 10, 12, 13, 15, 16}`, nine
joints — so no construction is flattered by being scored on joints it pins. The
all-seventeen figure is reported alongside, as elsewhere.

### Criteria, fixed before the numbers exist

1. **Sanity.** The `both` variant must reproduce the stored template-comparison
   ordering: template lower than anatomical on both backbones.
2. **Primary.** Some variant beats template Kabsch, paired cluster-bootstrap CI
   excluding zero, **on both backbones**. One backbone is not two.
3. **Secondary (reported regardless).** The gap between the best variant and
   Kabsch, against the published gap for `both`.

### Readings, fixed before the numbers exist

1. **A variant beats Kabsch on both backbones.** The published conclusion was an
   artifact of frame choice and must be corrected in the abstract, the
   contributions and the conclusion. This is the outcome that would change the
   thesis.
2. **The best variant narrows the gap but does not close it on both backbones.**
   The conclusion stands with a corrected, smaller margin, and the report states
   that the comparison was re-run against the best available construction.
3. **No variant improves on `both` against Kabsch.** The conclusion stands
   unchanged and the gap is confirmed not to be a frame-choice artifact.

Readings 2 and 3 change no existing number; they add one table and one sentence.

---

## Experiment 16 — asymmetric corruption

### Why this exists

The mechanism established by the tenth and eleventh experiments: Kabsch fits a
**rotation**, and bilaterally **symmetric** perturbations leave the point
cloud's principal axes where they are, so the fit barely moves. Both previous
corruptions were symmetric — all eight distal joints, or both limbs rescaled.

An **asymmetric** corruption should rotate the fit. Corrupting only the left
distal joints `{5 l_knee, 6 l_foot, 12 l_elbow, 13 l_wrist}` displaces mass on
one side only, while the anatomical frame reads `{0, 1, 4, 8}` and is untouched.

**This hypothesis has no literature support.** A 104-agent sweep looking for any
documented regime where a landmark frame beats Procrustes under unilateral
failure returned zero surviving claims. It is run because the mechanism is
sound, not because prior work suggests it.

### Design

Identical to `evaluation/occlusion_robustness.py`, including its scored set
`{9, 10, 11, 14}` — uncorrupted and non-constructor — so the two are directly
comparable. Only the corrupted set changes, from eight bilateral joints to four
left-side joints. Severities σ ∈ {0, 20, 40, 80, 160} mm, independent per
camera, both backbones. Arm C (template Kabsch on the four constructors) is
retained as the control.

### Criteria, fixed before the numbers exist

1. **Sanity.** At σ = 0, template lower than anatomical on both backbones.
2. **Primary.** Anatomical lower than template-17, CI excluding zero, at some
   **σ ≤ 80 mm on both backbones** — the identical threshold the tenth
   pre-registration used and failed, so the two are comparable.
3. **Comparison.** The crossover severity against the tenth experiment's. If
   asymmetric corruption crosses over earlier than symmetric, the mechanism is
   confirmed even where criterion 2 fails.

### Readings, fixed before the numbers exist

1. **Crossover at σ ≤ 80 on both backbones.** Asymmetric failure is the regime
   the previous four searches missed. Reported with the mechanism and with the
   explicit caveat that injected one-sided noise is not the same as real
   unilateral occlusion or pathological gait.
2. **Crossover on one backbone, or only above 80 mm.** Fails, exactly as the
   tenth did. Reported as the sixth failure of this family, with the crossover
   comparison to the tenth as a descriptive observation.
3. **No crossover.** The asymmetry mechanism is wrong and the symmetric
   explanation for Kabsch's robustness is incomplete. Reported as such.

---

## Isolation and cutoff

New modules `evaluation/best_frame_baseline.py` and
`evaluation/asymmetric_corruption.py`; new artifact directories
`thesis_artifacts/bestframe/` and `thesis_artifacts/asymmetric/`. No file
producing an audited number is modified.

**Both are abandoned if not complete by 8 Aug 2026, 20:00.** The defence is
9 Aug and the report is complete and internally consistent without either.
