# Prompt for an external AI review

Paste everything below the line into a fresh conversation with a different
model, and attach `Full_Thesis_Report.pdf`. Do not attach this file's context or
any of my earlier analysis — the point is an independent read.

**Attach:** `Full_Thesis_Report.pdf` (90 pages). Optionally
`thesis_artifacts/**/PREREGISTRATION.md` (nine files) if the reviewer asks to
verify the pre-registration claim.

---

You are reviewing an undergraduate final-year thesis in Computer Science
(Comilla University, Bangladesh). The defence is in three days and the report is
**frozen** — I am not looking for encouragement, and I am not going to rewrite
the science. I want to know what an examiner will attack, so I can prepare an
answer rather than be surprised.

**Be adversarial. If you find nothing seriously wrong, say so plainly and
briefly rather than inventing minor issues to seem thorough — but understand
that "this is impressive work" with a list of compliments is a failed review and
tells me nothing I can use.** Where you disagree with the thesis, say so
directly. Where the thesis is right and I am wrong to worry, say that too.

## What the thesis does

A frozen monocular 3D pose estimator (MotionAGFormer-XS, 2.24M parameters;
MotionBERT/DSTformer, 42.5M) predicts skeletons in the observing camera's frame,
so the same motion filmed from two viewpoints yields two different coordinate
sets. The thesis builds a body-fixed orthonormal frame from anatomical axes
(torso and hips) by Gram-Schmidt and applies it *after* prediction — no
retraining, no calibration, no labels, no new parameters.

It then asks what governs whether such a frame is consistent across viewpoints,
and tests that at three levels with **seventeen pre-registered experiments**
across sixteen documents, each criterion committed to version control before the
run.

Headline: on Human3.6M, which played no part in developing the method,
canonicalization cuts cross-view distance by **72.2%** over 180 held-out pairs
(179 improve), measured on the thirteen joints the frame is *not* built from.
Over all seventeen it is 74.1%.

## What the thesis already admits — do not spend your review rediscovering these

The report states all of the following. Repeating them back to me is not a
finding:

1. **The frame construction is not novel.** It is the TRIAD algorithm (Black,
   1964, spacecraft attitude determination). The rule that the better-determined
   direction should be primary is Shuster and Oh (1981). Both are cited and
   credited.
2. **The error propagation is not novel.** The relation that a direction from two
   joints distance L apart is uncertain by ~2σ/L restates Della Croce, Cappozzo
   et al. (1999, 2005) on landmark misplacement in biomechanics. An earlier draft
   claimed it as a derivation; the report now retracts that in the text.
3. **The closest precedent is named.** Wei, Lan, Zeng and Chen (2019) transform
   predicted poses to consistent views at the global body level and then at the
   level of local body parts — the same two-level decomposition — and the report
   says so explicitly.
4. **A simpler baseline beats the method.** Kabsch alignment to a single fixed
   reference skeleton is training-free, label-free, calibration-free and
   single-view, and it wins on **all 180 pairs under both backbones** and on
   **all fifteen actions**. Stated in the abstract, contributions, §5.10,
   Limitations and the conclusion.
5. **More than half of the seventeen pre-registered experiments failed their own criteria**, and
   a sixth returned the competing method as better. Reported as failures.
6. **The reliability score is falsified as an accuracy predictor five independent
   ways.** It is retained because its failure is a finding, and because it does
   gate canonicalization quality.
7. **A bone-length signal reaching ρ=0.492 on one dataset falls to ρ=0.098 on the
   other and is retracted.**
8. **The 55.1% multi-scale figure is circular** — each limb frame is built from
   the three joints it is scored on — and has been demoted to exploratory.

## What I actually want from you

Work through these in order. Be specific: cite page or section numbers.

1. **Correctness.** Any statistical, geometric or logical error. The bootstrap is
   a cluster bootstrap over 30 subject-action groups, 10,000 draws. Two averaging
   conventions (per-pair mean vs ratio-of-aggregate-means) are disclosed as
   disagreeing. Is that handled correctly, and is the disclosure adequate?
2. **Circularity.** §5.14 and the multi-scale demotion exist because a fit/score
   overlap was caught once. **Is there another one still in the report that was
   missed?** This is my single largest worry.
3. **Surviving overclaims.** Given items 1–8 above are already conceded, is
   anything *still* claimed more strongly than the evidence supports? Quote the
   sentence.
4. **Underclaims.** Anything genuinely defensible that the report has talked
   itself out of? Excessive hedging is also a defect.
5. **Prior art I have missed.** Especially 2024–2026 work on training-free
   canonicalization, view-invariant pose representations, or anatomical frames
   for pose comparison. If something out there already does this, I would rather
   know now than at the defence.
6. **Examiner simulation.** Write the five hardest questions a viva panel will
   ask, in the order they will ask them, and mark which of the five I would
   struggle to answer given only what is in the report.
7. **Is the contribution sufficient for an undergraduate thesis?** Answer
   honestly. The claimed contribution is not the method but the *experimental
   boundary*: axis length governs the choice between frame constructions, and
   governs neither which frame to trust within a construction nor which joint
   will disagree, because articulation dominates. Is that a real contribution or
   is it a negative result dressed up?

## The defence I currently plan for the baseline result

I want you to attack this specifically, because it is the argument the whole
defence rests on:

> The Kabsch-to-template baseline wins on the metric, but it **cannot run the
> experiment this thesis is about**. It has no anatomical axis, so there is no
> variable to hold fixed and vary, and the question of what governs frame
> consistency cannot be posed inside it. The anatomical frame is the instrument
> that makes the boundary measurable, not the result.

Is that sound, or is it special pleading? If an examiner says "you have built a
worse method and then redefined the contribution to be the fact that you
measured it", what is the strongest version of that objection, and does the
thesis survive it?

## Output format

- Findings ranked by severity, worst first.
- For each: the claim, where it appears, why it is wrong or weak, and what a
  three-day fix would be — or "no fix, prepare an answer instead", which is the
  realistic option for most of them.
- End with one paragraph: if you were the external examiner, what mark band would
  you place this in, and what single change would move it up one band?

Do not summarise the thesis back to me. I wrote it.
