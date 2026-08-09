# Defense brief — 10 August

Report is submitted. Nothing here changes the document. This is for the room.

**How to use it.** Tonight: read Parts 1–3 aloud, twice (~70 min), then Part 4
once (~30 min). Sleep. Tomorrow morning: Parts 1 and 5 only, 20 minutes.

---

## Part 1 — The first 60 seconds

Every reviewer of this work converged on one prediction: you will be asked why
your method exists when a simpler one beats it. So say it before you are asked.
Concede first, then reframe. Concede-then-reframe is much harder to attack than
reframe-alone, because there is nothing left to catch you hiding.

> "Sir, the problem is that the same person seen by two cameras produces two
> different sets of 3D coordinates, because each prediction lives in that
> camera's own frame. I kept MotionAGFormer-XS completely frozen and added one
> step after it: a coordinate frame built from the person's own hips and torso.
> Because that frame is built from the body, it rotates with the body, so the
> unknown camera rotation cancels — no calibration needed.
>
> The estimator's accuracy does not change and cannot: 45.149 mm before and
> after. What changes is agreement between cameras, from 372.7 mm to 93.4 mm,
> improving 179 of 180 held-out pairs.
>
> I should say the next part myself. A simpler baseline — Kabsch alignment to one
> fixed reference skeleton — reaches 57.5 mm where mine reaches 93.4, and beats
> mine on every pair. It is in my abstract. So I do not claim my method is the
> best alignment algorithm. What the thesis establishes is where this geometric
> reasoning holds and where it stops."

Then **stop talking.** Let him ask.

---

## Part 2 — The spine: four claims

If you can state these four in order, you can rebuild the whole thesis live.

| # | Claim | Evidence |
|---|---|---|
| 1 | Frozen predictions can be made cross-view comparable with no training, labels or calibration | 372.7 → 93.4 mm, 179/180 pairs; 75.8% on a second backbone |
| 2 | Axis length governs the choice **between** frame constructions | Longer shoulder axis beats hip axis, 5.2% and 4.4%, both backbones, intervals exclude zero |
| 3 | That principle has a boundary — it does **not** extend further | Frame-level: fails (circular). Joint-level: fails, slope 0.218 vs predicted band [0.038, 0.073] |
| 4 | It is not the best alignment algorithm, and I say so | Kabsch 57.5 vs 93.4, all 180 pairs, both backbones, 15/15 actions |

**The three-level story is the thesis.** Construction-level works. Frame-level
fails. Joint-level fails. Two of the three are failures, and the boundary is the
contribution.

---

## Part 3 — The three attacks, word for word

### Attack 1 — "Kabsch beats you. Why keep your method?"

Concede fully, then give the structural reason. Do **not** lead with the
structural reason; it sounds like moving the goalposts.

> "If the objective is only to minimise cross-view distance on this dataset, then
> Kabsch is better, and I report that in the abstract rather than a footnote. My
> objective became a different one: what determines whether an *anatomical* frame
> stays consistent? Kabsch fits a rotation to a point cloud — it has no anatomical
> axis, so there is nothing to hold fixed and vary. It cannot answer that question.
> So I am not claiming my method is the best aligner. I am claiming I measured
> where this kind of geometric reasoning carries and where it stops."

**If pushed — "then why does the frame matter at all?"**

> "Two things it gives that Kabsch does not. It needs no reference skeleton, only
> that the subject has hips and a torso. And its failure mode is disjoint from
> Kabsch's: corrupt the distal joints and my frame is unaffected at 53.45 mm while
> Kabsch degrades to 92.3; corrupt the frame's own anchor joints and mine collapses
> to 337.9 while Kabsch holds at 63.2. That disjointness is measured, on both
> backbones, and it is the one finding that is not conditional."

### Attack 2 — "Agreement is not correctness."

This is the strongest scientific objection. Agree with it immediately.

> "That is correct, Sir, and it is a real limitation — two predictions wrong in
> the same way agree perfectly. That is exactly why I keep MPJPE separate and never
> claim canonicalization improves accuracy. It cannot: the estimator is frozen.
> I bound the concern two ways: a per-frame Procrustes oracle that uses both views
> gives the floor any single-view method could reach, and I report how much of that
> gap is closed. I also tested whether the improvement buys a downstream task —
> cross-view retrieval — and recall fell. I withdrew that claim. So the honest
> position is that I improved a geometric property and have not demonstrated that
> it buys a downstream gain."

That last sentence is not a weakness. Volunteering it is what makes the rest
credible.

### Attack 3 — "Your table says 372.7 to 93.4. That's 74.9%, not 72.2%."

Somebody with a calculator will find this. The submitted report does **not** explain
it, so you must. It is a good answer — you reported the number that flatters you
less.

> "Both are correct, Sir, and they are different conventions. 72.2 percent is the
> mean of the per-pair improvements — I compute the improvement for each of the 180
> pairs and average those. 74.9 percent is the ratio of the two mean distances. The
> mean of ratios is not the ratio of means, so they differ by about two and a half
> points. I report the per-pair mean because it is the conservative of the two, and
> the bootstrap interval is computed on that same per-pair quantity over the thirty
> subject-action clusters. The stored artifact records both, so the choice is
> auditable."

Same at seventeen joints: 320.4 to 75.3 is 76.5 percent by ratio; I report 74.1.

### Attack 4 — "TRIAD is from 1964. What is novel?"

Memorise this one.

> "I would not claim the frame construction or the error-propagation principle is
> new. TRIAD is Black, 1964; the primary-axis rule is Shuster and Oh; the
> propagation of landmark error into frame orientation is established in
> biomechanics, and I cite all of it. My contribution is the experimental
> characterisation of how far that reasoning transfers to joints produced by a
> frozen neural network. It transfers at one level — choosing between frame
> constructions. It fails at two — predicting which frame instance is reliable, and
> predicting which joint will disagree, because articulation dominates there. That
> boundary is what did not exist before, and establishing it took seventeen
> pre-registered experiments."

---

## Part 4 — Depth drills

Sir said the report lacks depth. Depth in a viva means answering the *second* and
*third* "why", not the first. Practise these out loud.

**Why does the camera rotation cancel?**
Both axes are read from the joints. If camera B sees the same pose rotated by an
unknown `Q`, then `P⁽ᴮ⁾ = P⁽ᴬ⁾Q`, and the frame built from those joints rotates
with them, so `R⁽ᴮ⁾ = Qᵀ R⁽ᴬ⁾`. Then
`P⁽ᴮ⁾R⁽ᴮ⁾ = (P⁽ᴬ⁾Q)(QᵀR⁽ᴬ⁾) = P⁽ᴬ⁾R⁽ᴬ⁾`. `Q` never has to be estimated — that is
why no calibration is needed. *Third why:* it is exact only if both cameras give
the same pose up to rotation; in practice each carries its own error, and the
residual 93.4 mm **is** that error.

**Why should the longer axis be primary?**
A direction read from two joints `L` apart, each noisy by `σ`, has angular error
≈ `2σ/L`. Longer baseline, smaller angular error. The shoulder axis is longer than
the hip axis, so it makes the better frame — 5.2% and 4.4%, both backbones.

**Why does the same reasoning fail at joint level?**
The radius model predicts a joint at radius `r` is displaced by `rθ`. Measured
slope 0.218 against a pre-registered band of [0.038, 0.073]. The control explains
it: torso-rigid joints sit at a *larger* mean radius than articulated joints yet
disagree about 2.5× *less*. Shoulder and knee are within 1–3% in radius but differ
about twofold in canonical distance. So articulation, not radius, dominates.

**Why did the reliability score fail?**
Every component is computed from one frame's geometry. A pose that is wrong purely
in depth is still symmetric, correctly proportioned and well conditioned. Geometry
cannot see the error because the error is what geometry preserves.

**Careful here — it does not explain all four.** It explains the two *error
predictors*, the reliability score and the bone-length signal, which both asked a
geometric quantity to predict a non-geometric failure. The other two failed for
different reasons: TRIAD was a literature finding, not an experimental failure, and
the multi-scale variant was circular. If you claim one cause for all four, an
examiner who separates them has caught you.

**Why is n = 2 enough?**
It is not, and I say so. Two datasets and two backbones is replication, not
universality, and the report states every backbone-dependent conclusion as `n = 2`.
The reliability score behaved differently on a third architecture — the sign flipped
on a plain MLP — which is part of why it is reported as falsified.

---

## Part 5 — Numbers, verified tonight

Re-checked at 21:50 on 9 August: **304/304 claims, 76 tests, 16/16 pre-registrations
in order.**

| | |
|---|---|
| MPJPE, before and after | 45.149 mm — unchanged |
| Cross-view, 13 non-constructor joints | 372.7 → 93.4 mm (−72.2%), 179/180 |
| All 17 joints | 320.4 → 75.3 mm (−74.1%) |
| Second backbone (MotionBERT) | −75.8%, 180/180 |
| Kabsch baseline | 57.5 mm — beats 93.4 on every pair |
| Procrustes oracle floor | 56.2 mm; 87.0% of the gap closed |
| Distal corruption | frame flat 53.45 mm · Kabsch 43.3 → 92.3 |
| Anchor corruption | frame 53.45 → 337.9 · Kabsch 43.3 → 63.2 |
| Axis length | shoulder beats hip, 5.2% / 4.4% |
| Radial law | slope 0.218 vs band [0.038, 0.073] |
| Bone-length signal | ρ +0.492 → +0.098 — retracted |
| Experiments | 17 pre-registered in 16 documents; more than half failed |
| Cost | 0 parameters, 402 FLOPs/frame |

Where things live: **Table 5.1 p18** (every claim + verdict) · **Table 5.2 p22**
(base vs proposed) · **§5.3 p23** (the baseline that wins) · **Fig 3.1 p9**
(pipeline) · **Fig 3.2 p10** (frame construction) · **Fig 5.4 p21** (eight cameras).

---

## Part 6 — Do not say

- **Never cite an AI review.** No "Gemini rated it 9.6", no scores, no comparison
  with other students' theses. It adds nothing scientific and reads badly.
- **Never say "no training"** as shorthand. Say **"adds zero trained parameters."**
  The shorthand invites the objection that YOLOv8 and MotionAGFormer are trained.
- **Do not present the routing rule as an established contribution.** It is
  exploratory: the confidence signal is simulated, and Experiment 14 showed the
  real detector channel carries no usable signal. If asked whether Kabsch and the
  frame can be combined: *"Conceptually yes, and I measured it — but under a
  simulated confidence gate, so I report it as exploratory, not as a method."*
- **Do not quote a number you have not seen in the submitted PDF.** If unsure, say
  "I would have to check the table" — that is a perfectly good answer and far
  better than a wrong figure in a thesis whose whole claim is auditability.

---

## Part 7 — If you blank

Say: **"The contribution is the boundary, not the method."**

It is true, it is the thesis, and it buys you ten seconds to find the thread.

---

## On "it lacks depth"

If Sir raises it, do not argue. The honest answer is that the submitted version was
deliberately condensed, and the full technical treatment exists:

> "The submitted report is the condensed version, Sir. There is a 97-page technical
> version with the implementation, the joint conventions, the full evaluation
> protocol and the reproducibility infrastructure written out. I will send it this
> week."

Then send the supplement. It is already written, compiled and audited — it is not a
promise you have to keep by working.
