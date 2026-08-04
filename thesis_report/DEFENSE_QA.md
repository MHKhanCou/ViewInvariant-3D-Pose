# Defense Q&A

Defense: 9 August 2026. Read this the night before and the morning of.

Every number here is verified by `python -m evaluation.audit_numbers`
(101 claims). If a question asks for a figure not in this document, say you will
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

> I falsified it as a predictor of accuracy, along seven independent axes, and I
> report every one. It is retained as what it demonstrably is: a gate on
> degenerate and corrupted geometry, where it works — the correlation with
> induced corruption is −0.813, and it abstains on 100 percent of joint-dropout
> cases.
>
> The distinction the report draws is between detecting that a skeleton is
> malformed, which it does, and predicting that a well-formed skeleton is wrong
> in depth, which it cannot. A pose can be symmetric, correctly proportioned and
> well conditioned while being wrong, and that is precisely the failure mode a
> single-frame geometric score cannot see.

**Before the defense, email your supervisor**: say the reliability component was
falsified as an accuracy predictor and retained as a degeneracy gate, and ask
whether he wants the title adjusted. Raising it yourself converts a weakness
into evidence of rigour. Do not change an approved title unilaterally.

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
| Audit / tests | 101 claims, 97 tests |

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
