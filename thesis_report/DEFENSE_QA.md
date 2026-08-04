# Defense Q&A

Defense: 9 August 2026. Read this the night before and the morning of.

Every number here is verified by `python -m evaluation.audit_numbers`
(131 claims). If a question asks for a figure not in this document, say you will
check it rather than guess — you have never once quoted a number you could not
trace, and that is worth more than one recalled digit.

---

## The 30-second answer, if you only get one

> I make the predictions of a frozen 3D pose estimator comparable across camera
> viewpoints, by constructing a body-fixed coordinate frame from the predicted
> anatomy after prediction. It adds no trained parameters, needs no labels and
> no camera calibration. On Human3.6M, which I did not use to develop the
> method, it reduces cross-view distance by 74 percent across 180 held-out
> camera pairs and improves 179 of them.

Then stop talking. Let them ask.

---

## THE FIVE HARD QUESTIONS

### 1. "But it isn't training-free — you use pretrained networks."

**They are right, and the report says so on its first methodology page.**

> The pipeline is not training-free and Section 3.1 states that plainly. The
> detector and the lifting network are trained by their authors and used frozen.
> What is training-free is everything I add: zero trained parameters.
>
> The property I claim is a property of the deployment requirement, not of the
> pipeline's history. To apply my framework to a new estimator, a new camera or
> a new domain you need no training data, no labels, no calibration and no
> gradient step. To apply 3DPCNet, the closest prior work and also
> estimator-agnostic, you must train its network. Both sit downstream of a
> trained estimator. The difference is entirely in what the user must supply.

If pushed on what would falsify it:

> Not the presence of a pretrained backbone, which is assumed. It would be
> falsified if the method needed per-dataset tuning. The constants were fixed on
> MPI-INF-3DHP, never adjusted, and applied to 180 unseen Human3.6M pairs.

**Never say "no training" as shorthand. Say "adds zero trained parameters."**
The shorthand is what invites the question.

---

### 2. "What did you actually invent? Canonicalization already exists."

Concede immediately — the report already does, in Section 2.3.

> Body-frame canonicalization is not novel and I say so in the literature
> review. 3DPCNet addresses the same problem and reports a hand-built baseline
> of this kind performing worse than its learned alternative.
>
> What is mine is the requirement profile and the evidence. Zero trained
> parameters, no labels, no calibration, evaluated on 209 camera pairs across
> two datasets, reaching 90.5 percent of a Procrustes oracle. 3DPCNet reported
> that a hand-built frame underperforms. My data says that is a function of
> which axes you build it from.

**If they push harder, this is your strongest card:**

> The axis-length result. I predicted from the construction that a frame's
> sensitivity to joint error scales inversely with the length of the axis it is
> built from. I then found a bilateral asymmetry I was not looking for: the two
> legs differed by a factor of 2.3 while the arms agreed. Tracing it, the legs
> had been defined with different axes. Correcting that raised the multi-scale
> improvement from 25.6 to 37.2 percent. I then predicted the same defect in
> both arms, which the bilateral check could not reveal because both arms shared
> it, and confirmed it: 55.1 percent, all 180 pairs improving, and all five
> levels converging into a 4.4 mm band where they had spanned 28.6 to 69.4.
>
> That convergence was not optimised for. One rule applied to five anatomically
> different segments produced five similar numbers. That is a design principle
> any landmark-based frame construction inherits.

---

### 3. "You never improved accuracy — you only changed the representation."

Two things wrong with the premise. Say both.

> Improving accuracy was never the claim, and that is deliberate. The objective
> is cross-view comparability. Criticising the method for not improving MPJPE is
> like criticising a compression format for not improving image quality.
>
> And it is not quite true. Fusion does improve accuracy: 37.8 mm to 34.6 mm
> across four uncalibrated views, with a bootstrap interval of [+2.1, +13.7]
> that excludes zero.

---

### 4. "Your title says Reliability-Aware, but you falsified the reliability score."

Do not be defensive. This is your best evidence of rigour.

> I falsified it as a predictor of accuracy, along five independent axes, and I
> report every one. Then I tested it against the target its own specification
> names — whether the body frame is fit to canonicalize with — and against that
> target it works, on both backbones. Discarding the worst thirty percent of
> frames by reliability lowers mean cross-view distance from 76.2 to 59.5 mm,
> where discarding a random thirty percent changes nothing.
>
> So the distinction is between predicting that a pose is wrong in depth, which
> it cannot do, and predicting that a frame is unfit to build, which it can. A
> pose can be symmetric, correctly proportioned and well conditioned while being
> wrong in depth — that is the failure mode a single-frame geometric score
> cannot see. It is not the failure mode that breaks canonicalization.

**If they push on whether this is post-hoc rescue, concede the shape of it and
give the controls.** This is the honest answer and it is strong enough:

> It is exploratory, and I say so in the report. It was a comparator in a
> pre-registration whose actual subject — a geometric conditioning index —
> failed. What makes me willing to state it is that it replicates on two
> backbones with intervals excluding zero, +5.44 mm [+1.53, +9.27] and
> +2.74 mm [+1.55, +3.81], and survives a partial correlation controlling for
> bone-ratio deviation at −0.306 and −0.322, so it is not just detecting
> distorted skeletons. My bone-length signal had less than that and it still
> died on the second dataset, so I state this one as exploratory and no more.

**Before the defense, email your supervisor**: say the reliability component was
falsified as an accuracy predictor and then supported as a gate on
canonicalization quality, and ask whether he wants the title adjusted. Raising
it yourself converts a weakness into evidence of rigour. Do not change an
approved title unilaterally.

---

### 4b. "Isn't your headline metric just measuring that two wrong answers agree?"

**This is the sharpest question available and it is not fully answerable.** Do
not bluff. Give the three defences and then concede the gap.

> Three things guard against it. The Procrustes oracle is the floor — it aligns
> the two predictions optimally with full knowledge of both, so no rotation-based
> method can beat it, and canonicalization closes 90.5 percent of the gap to it.
> The multi-scale distance keeps its correlation with ground-truth error, +0.610
> against +0.601 for the global frame, so collapsing agreement is not what
> produced the gain. And the rotation-cancellation result is exact, so the
> mechanism is not statistical.
>
> What I cannot claim is that reducing this metric improves a downstream task.
> My retrieval experiment is negative and used a superseded protocol. That is
> the clearest gap in the work and it is written as such in the future work
> section.

---

### 5. "If the Procrustes oracle is better, why not just use it?"

> The oracle aligns two poses to each other, so it needs both at once, and what
> it returns is a relative alignment rather than a representation. It cannot
> canonicalize a single pose, because there is nothing to align it to. It cannot
> run before a comparison exists, so it cannot index, store or stream. With N
> cameras it must be recomputed for every one of the N(N−1)/2 pairs, whereas a
> body frame is computed once per pose and every pair is then comparable.
>
> It is a floor on what any rotation can achieve, not a competing method. My
> claim is that a construction seeing one pose at a time recovers 90.5 percent
> of what a construction seeing both can do.

---

### 6. "Your proposal said you would train a network with geometric losses. You didn't."

Your supervisor wrote the proposal. Expect this, possibly first. Do not be
defensive: the problem and the gap are unchanged, only the method inverted.

> The problem statement and the research gap are exactly the ones in the
> proposal: existing methods lack explicit geometric priors and view-invariant
> constraints. What changed is the method. The proposal was to learn
> view-invariance through a loss, with a view-invariance term and a bone-length
> term added to the pose loss. I derived it instead.
>
> The reason is that the rotation cancels algebraically. If two cameras see one
> pose, their predictions differ by an unknown rotation, and because the frame is
> built from the joints themselves it rotates with them, so the unknown rotation
> cancels exactly and never has to be estimated. Once that is true, training is
> not required to obtain the invariance, and the survey showed the learned route
> was already occupied by 3DPCNet, MoViD, V-VIPE and CanonPose. The
> requirement profile — no training, no labels, no calibration — was the part
> nobody had taken.

**Then volunteer the honest accounting, because it is your strongest move:**

> The proposal hypothesised that explicit geometric priors would deliver both
> view-invariance and reliability. I tested that hypothesis in its cheapest
> possible form, analytically and with zero training, and the verdict is split.
> The view-invariance half holds: 74.1 percent across 180 held-out pairs on a
> dataset I did not develop on. The geometric-priors-as-quality-signal half does
> not: bone-length consistency scored +0.492 on the first dataset and +0.098 on
> the second, and I retract it. So the proposal's question is answered, not
> avoided, and half the answer is negative.

**What was not done, if asked:** CMU Panoptic and CASIA Gait were not used, and
no gait or sports application was evaluated. The proposal listed four datasets;
two were used, both multi-camera, which is what the central claim requires. Say
this plainly rather than let it be discovered.

---

## THE RETRACTIONS — lead with them, do not wait to be asked

Volunteering these is the single most effective thing you can do. It makes
everything else you say more credible.

### The bone-length signal

> My strongest single-view result was a temporal bone-length signal at ρ = +0.492
> on MPI-INF-3DHP. I tested it on Human3.6M and it collapsed to +0.098, failing
> all five criteria. I retract the general claim.
>
> The result is trustworthy because the apparatus is verified: my pipeline
> reproduces the backbone's published accuracy to three decimal places, 45.149
> against 45.149 from its own evaluation script. And I tested the convenient
> excuse — that Human3.6M is in-domain so there is little error to predict — by
> comparing per-video correlation against per-video error. Flat. The hardest
> third of videos gives +0.156, the easiest +0.136.
>
> What survives is narrower: the signal detects gross skeletal deformation from
> severe domain shift, not residual depth error. Median aligned error is 202 mm
> on one dataset and 33 mm on the other.

**If asked why it failed:** the same mechanism that defeated the reliability
score, extended along time. Geometric plausibility is invariant to a coherent
depth error — within a frame, and across frames when the error is temporally
coherent.

### The SittingDown mechanism

> I originally wrote that SittingDown fails because a seated pelvis shortens the
> hip axis. Building a figure to show it, I measured it, and it is false.
> SittingDown has the second longest hip axis of any action at 285.3 mm; the
> shortest is WalkDog at 268.4 mm, which works fine. Canonical distance
> correlates with hip-axis length at −0.03 and with the backbone's own accuracy
> at +0.76, which survives deleting SittingDown at +0.70.
>
> So the failure is inherited from the estimator, not produced by the frame.
> That is a less comfortable conclusion, because no better frame construction
> repairs it.

---

## NUMBERS TO KNOW COLD

| Claim | Number |
|---|---|
| Cross-view, MPI-INF-3DHP | +32.4%, 27 held-out pairs |
| Cross-view, Human3.6M | **+74.1%**, CI [+69.8, +77.2], 179/180 pairs |
| Oracle gap closed | 90.5% |
| Frame validity | 100% |
| Multi-scale, as implemented | +25.6% |
| Multi-scale, own long axis | **+55.1%**, CI [+53.6, +56.5], 180/180 |
| Fusion, median over 4 views | 37.8 → 34.6 mm, CI [+2.1, +13.7] |
| Fusion, unweighted mean | −3.4%, CI spans zero — **not** reliably worse |
| Bone signal | +0.492 → **+0.098**, retracted |
| Reliability vs corruption | −0.813 (this one works) |
| Backbone reproduction | 45.149 mm vs 45.149 published |
| Added trainable parameters | **0** |
| Canonicalization cost | 402 FLOPs/frame, 0.0005% of backbone |
| Audit / tests | 131 claims, 67 tests |

**Two datasets: 209 camera pairs total. Four subjects.**

---

## IF YOU DO NOT KNOW

> I do not have that number in my head. Every claim in the report is recomputed
> from stored artifacts by an automated audit, so I can give you the exact figure
> from the source rather than guess.

This is a strong answer, not a weak one. Use it without embarrassment.

---

## WHAT NOT TO DO

- Do not oversell. The report's credibility comes from its retractions.
- Do not say "novel" about the canonicalization. Say "the requirement profile".
- Do not defend the reliability score as an accuracy predictor. You disproved it.
- Do not quote +0.492 without immediately saying it did not replicate.
- Do not claim you improved MPJPE, except for fusion, where you did.
- Do not say "no training". Say "adds zero trained parameters".

## WHAT TO DO

- Open with Figure 5.x: four cameras superimposed, raw versus canonical. It
  explains the thesis in ten seconds, before you say anything.
- Volunteer the retractions before you are asked.
- When you state a number, say where it came from.
- If a question rests on a false premise, correct the premise first, politely.
