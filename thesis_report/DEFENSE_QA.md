# Defense Q&A

Defense: 9 August 2026. Read this the night before and the morning of.

Every number here is verified by `python -m evaluation.audit_numbers`
(304 claims). If a question asks for a figure not in this document, say you will
check it rather than guess — you have never once quoted a number you could not
trace, and that is worth more than one recalled digit.

---

## FOUR LENGTHS — rehearse each aloud until it is not a recitation

### 10 seconds

> Two cameras looking at the same person produce two different sets of 3D
> coordinates. I remove the camera from the answer using the body's own anatomy,
> with no training and no calibration — and I establish exactly how far that
> geometric reasoning carries.

### 30 seconds

> Monocular 3D pose estimators report the skeleton in the camera's frame, so the
> same pose from two viewpoints gives two different answers. I build a coordinate
> frame out of the predicted body itself — torso axis, then hip axis — and apply
> it after prediction. It adds no parameters and needs no calibration. On
> Human3.6M, which I did not use to develop it, cross-view distance falls 72.2
> percent across 180 held-out pairs and 179 of them improve. The construction is
> the TRIAD algorithm from spacecraft attitude determination; I do not claim it.
> What I contribute is where the reasoning behind it stops working on a human
> body.

### 2 minutes

> Add, after the 30-second version:
>
> The design question is what to build the frame from, and the geometry answers
> it: a direction read from two joints a distance L apart is uncertain by about
> 2σ/L, so use the longest axis available. The clean test of that is a
> global-frame comparison: swapping the hip axis for the longer shoulder axis,
> with the scored set and the constructor count held identical, wins on both
> backbones by 5.2 and 4.4 percent with intervals excluding zero.
>
> I then pre-registered three tests of how far that principle goes, committing
> each criterion to version control before running. It decides between frame
> constructions — established by the controlled global-frame comparison above,
> not by the limb-level correlation, which the section "A Circularity Control, and a Demotion" shows is confounded with
> constructor count. It
> does **not** decide which individual frame to trust, because within one
> construction the axis is essentially the subject's hip width and barely varies.
> And it does not decide which joint will disagree, which is the interesting
> failure: joints rigidly attached to the torso sit *further* from the root than
> the knees and yet disagree two and a half times *less*. That contradicts the
> rigid-body prediction in the wrong direction to be a matter of degree. A body
> has hinges; a spacecraft does not.
>
> Two of my three pre-registered tests failed, and the boundary they establish is
> the contribution.

### 5 minutes

> The 2-minute version, then the negative results as a block, in this order,
> because volunteering them is worth more than defending them:
>
> 1. **The reliability score.** I proposed it, then falsified it as an accuracy
>    predictor five independent ways, and identified the cause — a within-frame
>    plausibility measure is blind to a coherent depth error, which is what
>    dominates monocular estimation. Then I tested it against the target its own
>    docstring names, whether the frame is fit to canonicalize with, and it works
>    on both backbones. Five careful falsifications had all been aimed at the
>    wrong target. I report that as exploratory because it was a comparator in
>    its pre-registration, not its subject.
> 2. **The bone-length signal.** My strongest single-view finding, ρ = +0.49 on
>    MPI-INF-3DHP. It falls to +0.10 on Human3.6M and fails every criterion. I
>    retract it and state the narrower condition where it holds.
> 3. **Redundant landmarks.** Combining more axes made things worse. The
>    explanation is Wahba's problem: least-squares over multiple directions is
>    optimal only under inverse-variance weighting and independence, and both
>    fail here — my axes span 138 to 461 mm, and one network predicting all
>    seventeen joints makes their errors dependent.
>
> Then the apparatus: 304 numerical claims recomputed from stored artifacts by an
> automated audit, 76 tests, seventeen pre-registered experiments timestamped ahead of their
> results, and two claims withdrawn during the final week because they did not
> survive a check I should have run earlier.

---

## The 30-second answer, if you only get one

> I make the predictions of a frozen 3D pose estimator comparable across camera
> viewpoints, by constructing a body-fixed coordinate frame from the predicted
> anatomy after prediction. It adds no trained parameters, needs no labels and
> no camera calibration. On Human3.6M, which I did not use to develop the
> method, it reduces cross-view distance by 72.2 percent across 180 held-out
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
> two datasets, reaching 87.0 percent of a Procrustes oracle. 3DPCNet reported
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
> it, and confirmed the defect was there. **But I then found the measurement
> was largely circular** -- each corrected limb frame is built from exactly the
> three joints it is scored on, and the torso from four of its six -- so I
> demoted that result to exploratory and it is not evidence for anything. The
> convergence confirms the implementation fix was applied uniformly, nothing
> more.
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

### 4. "You built a reliability score and then falsified it. Why is it still in the thesis?"

**First, the title.** "Reliability-Aware" is no longer in it — it was removed
because the evidence did not support that prominence. If he raises the old title
from an earlier draft, say so plainly and move to the science below. Volunteering
the removal is stronger than defending the word.

Then, do not be defensive. This is your best evidence of rigour.

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
canonicalization quality, and ask him to confirm the title. Raising it yourself
converts a weakness into evidence of rigour.

**If asked whether the title overclaims, it does not, and you should know why it
is worded as it is.** It says *canonicalization framework*, not pose estimation,
because the estimator is frozen and only the coordinate frame changes. It says
*cross-view comparability*, not view-invariant estimation, because what improves
is agreement between two camera-relative predictions and not their accuracy.
Those two choices pre-empt the most common objection to work of this kind.

---

### 4b. "Isn't your headline metric just measuring that two wrong answers agree?"

**This is the sharpest question available and it is not fully answerable.** Do
not bluff. Give the three defences and then concede the gap.

> Three things guard against it. The Procrustes oracle is the floor — it aligns
> the two predictions optimally with full knowledge of both, so no rotation-based
> method can beat it, and canonicalization closes 87.0 percent of the gap to it.
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
> claim is that a construction seeing one pose at a time recovers 87.0 percent
> of what a construction seeing both can do.

---

### 6. "Your proposal said you would train a network with geometric losses. You didn't."

Sir's brief set this direction, and your own submitted proposal committed to
it. Expect this, possibly first. Do not be
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
> The view-invariance half holds: 72.2 percent across 180 held-out pairs on a
> dataset I did not develop on, measured off the four joints the frame is built
> from; 74.1 percent if all seventeen are counted. The geometric-priors-as-quality-signal half does
> not: bone-length consistency scored +0.492 on the first dataset and +0.098 on
> the second, and I retract it. So the proposal's question is answered, not
> avoided, and half the answer is negative.

**What was not done, if asked:** CMU Panoptic and CASIA Gait were not used, and
no gait or sports application was evaluated. The proposal listed four datasets;
two were used, both multi-camera, which is what the central claim requires. Say
this plainly rather than let it be discovered.

---

### 7. "Isn't 2σ/L just Cappozzo? Isn't your frame just TRIAD?"

**Both yes. Say so before he finishes the sentence.** This is the question that
would have ended the defense a week ago and is now one of your strongest moments,
because you found it yourself and put it in Chapter 2.

> Yes to both, and Section 2.5 says so at length. The construction is TRIAD,
> Black 1964. The rule that the better-determined axis should be primary is
> Shuster and Oh, 1981. The propagation from landmark error to frame orientation
> is Della Croce and Cappozzo, 1999 and 2005. I claim none of it, and an earlier
> draft of my report did claim the propagation — I withdrew that during the final
> literature review.
>
> What I claim is two things. The transfer: that reasoning was built for markers
> placed by a human examiner, and I apply it where the landmarks are network
> outputs, whose error is dominated by coherent depth ambiguity rather than
> independent placement noise. And the boundary, which is the real result: the
> rigid-body pivot relation those papers rely on does not survive on an
> articulated skeleton, and I have the pre-registered experiment that shows it.

**If he presses — "so you found nothing new":**

> I found where a sixty-year-old piece of geometry stops applying to the problem
> everyone is now using it for, and I found it by pre-registering a test I
> expected to pass. That is a smaller claim than the one I started the year with,
> and it is the one that survives.

---

### 7c. "You used the favourable averaging convention where it suited you."

The sharpest statistical question available. **Concede the shape of it, then
give the reason** -- the report already discloses this, so he is reading your
own disclosure back to you.

> That asymmetry is real and it is in the report. The cross-view improvement
> uses the per-pair mean, which is the lower of the two -- 72.2 against 76.5 --
> and the fusion headline uses the ratio of aggregate means, which is the higher
> one. I report both in both places rather than choosing for the reader.
>
> The reason is the unit the experiment is defined on. An improvement defined
> per camera pair should be averaged per camera pair, or the widest-baseline
> pairs dominate. The fusion experiment is defined on pooled frames, so the
> pooled ratio is its natural unit. I did not pick per experiment to flatter the
> number -- and where the conservative convention changes the verdict, as it
> does for the median fusion, I report the weaker claim: +4.7 percent per frame
> with an interval spanning zero.

**Do not** say "the conservative convention is used throughout". It is not, and the report says so where the convention is first stated.
Cite the section by NAME at the viva, never by number.

---

### 7d. "One comparison on two backbones, and you call it 'governs'?"

He has conceded everything else and is squeezing the single surviving positive
claim. This is fair and you should not bluster.

> It is one controlled comparison, yes -- and it is the only one in the report
> where the scored set and the constructor count are held identical and only the
> axis length differs, which is why it is the one I cite and why I do not cite
> the limb-level correlations that the section "A Circularity Control, and a Demotion" shows are confounded.
>
> What it has is replication rather than volume: two independently trained
> networks, different architectures, nineteen-fold different size, byte-identical
> evaluation code, both intervals excluding zero, and the direction registered
> before the number existed. The effect is small -- 5.2 and 4.4 percent -- and I
> state it as such rather than rounding it up.
>
> If you are asking whether 5 percent is worth a contribution, my answer is that
> the contribution is the boundary rather than the effect size: this is the level
> where the principle still holds, and the two levels below it are where it stops.

---

### 7b. "Your circularity table leaves out the torso."

**He is right, and an earlier draft did.** The table now includes it. Answer:

> Yes -- the torso is in the demoted band with the limbs, and the table lists it
> at 1.22 times its floor. It is not a three-joint segment, but four of the six
> joints it is scored on build its frame, so the same objection applies. I had
> originally written that the circularity was confined to three-joint limbs,
> which was too narrow, and the section "A Circularity Control, and a Demotion" now says it is a property of any
> segment frame built mostly from the joints it is scored on.
>
> What it does not touch is the global frame, and the reason is the ratio rather
> than the segment size: four constructors against seventeen scored, thirteen
> joints held out, and it sits at 1.46 times its floor with real disagreement
> still measurable. The headline result is scored there.

**If he pushes: "so your five converging levels are all circular?"** Concede it
immediately -- this is the honest answer and the report now says it:

> All five, yes. Four limbs built from three of three, the torso from four of
> six. I previously called that convergence my strongest evidence for the
> axis-length principle and I withdrew it, because convergence is exactly what
> the circularity predicts by itself. The evidence for the principle is the
> global-frame comparison where only the axis length differs -- shoulder axis
> against hip axis, same scored set, same constructor count, wins on both
> backbones with intervals excluding zero.

---

### 8. "Your headline percentage doesn't match your own table."

He may divide 320.4 and 75.3 and get 76.5, not 74.1. **This is disclosed in
the section "Cross-View Canonicalization on Human3.6M"; know it cold.**

> Every improvement figure is the mean over camera pairs of each pair's own
> percentage, not the ratio of the aggregate means. The ratio of means would give
> 76.5 here and 81.0 on the second backbone — both larger. I report the smaller,
> per-pair convention because it weights every pair equally instead of letting
> the widest-baseline pairs dominate, and the section "Cross-View Canonicalization on Human3.6M" states this in the
> paragraph where the numbers first appear.

---

### 9. "You bootstrapped that constant the day before submission."

> Yes. The report had claimed that two backbones producing constants within one
> percent of each other showed the mechanism was physical. That inference needs a
> confidence interval on the constant and I had not computed one. When I did, the
> interval was wider than the estimate — 19 to 42 against a point estimate of 21
> — so the agreement was a coincidence. I withdrew the argument in the text
> rather than deleting it quietly, and the bootstrap now runs inside the audit.

**Do not apologise past this point.** Finding your own error late is better than
not finding it.

---

### 10. "Your triage result wasn't pre-registered." — WEAK, concede it

> Correct, and I label it exploratory everywhere it appears. It was the
> comparator in a pre-registration whose actual subject failed. What makes me
> willing to state it at all is that it replicates on two backbones with
> intervals excluding zero and survives a partial correlation controlling for
> skeletal distortion. My bone-length signal had less support than that and still
> died on the second dataset, which is exactly why I will not call this
> confirmed.

---

### 10b. "Distal joints having more error is common knowledge. What did you find?"

**He is right and you must agree immediately.** This is the newest correction and
the most recently checked.

> He is right, and I say so in the report. Error accumulating toward the distal
> end of a kinematic chain is standard biomechanics, and wrists and ankles
> carrying the largest error is visible in the per-joint table of every paper on
> this benchmark. I did not discover that.
>
> What I tested was a model, not the phenomenon. Having found that axis length
> governs the choice between frame constructions, the obvious next step is to
> carry the same geometric account down to individual joints — error should grow
> with radius. I pre-registered that, tested it parameter-free with constants
> measured in a different experiment, and it failed, with a matched-radius
> control ruling out the confound: the shoulder and the knee differ by 1.2
> percent in radius and a factor of 2.6 in disagreement. So the result closes a
> door that my own two previous sections leave open, and saves the next person
> the experiment.

**Do not say "novel" anywhere in this answer.**

---

### 10c. "A simpler method beats yours. Then why build the frame at all?"

**This follows immediately once they accept your framing, and it is the question
the whole thesis turns on. Know it cold.**

First concede without hedging:

> Kabsch alignment to a fixed skeleton beats my frame on both backbones, every
> centring, three unrelated templates and all fifteen actions. I found no pose
> regime where mine is preferable, I pre-registered that comparison with its
> readings fixed beforehand, and it is in my abstract.

Then the answer they are actually asking for:

> **Because the baseline cannot run the experiment.** It has no anatomical axis,
> so there is no variable to hold fixed and vary. My central result is that axis
> length decides between frame constructions and nothing finer, and the test for
> it is hip axis against shoulder axis with the joint set, the scored set and
> the constructor count held identical and only the length differing. That
> comparison does not exist in a template-alignment method — there is no axis to
> lengthen. The frame is the instrument because it exposes the parameter the
> question is about.
>
> So the baseline wins the engineering comparison and is silent on the
> scientific one. Section 2.6 called the construction an instrument rather than
> a result before I had any reason to need that framing, and the boundary
> analysis is unaffected by the baseline result.

**Do not present this as consolation.** It is a statement about what each method
can and cannot measure, and it is true independently of which one aligns better.

---

### 11. "Show me it's useful for anything." — WEAKEST ANSWER, do not bluff

> I can't, and it is the clearest gap in the work. My retrieval experiment is
> negative and used a protocol I later superseded. Every result in the report
> measures a geometric quantity, and none demonstrates that improving it improves
> a downstream task. That is stated in the limitations and it is the first thing
> I would do with more time.

Say it in that order — concede, explain, own it. Any attempt to dress this up
will be seen through, and the honest version costs you less.

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
> matches the backbone's own evaluation script to three decimal places, 45.149
> against 45.149, and the paper's published 45.1 to within 0.05 mm — the paper
> reports one decimal, so that is the closest a comparison to it can get. And I tested the convenient
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
| Cross-view, Human3.6M | **+72.2%** off the frame's own joints (+74.1% over all 17), 179/180 pairs |
| Oracle gap closed | 87.0% off the frame's own joints (90.5% over all 17) |
| Frame validity | 100% |
| Multi-scale, as implemented | +25.6% |
| Multi-scale, own long axis | +55.1% -- **DEMOTED, largely circular (the section "A Circularity Control"). Do not volunteer this number.** |
| Fusion, median over 4 views | 37.8 → 34.6 mm, CI [+2.1, +13.7] |
| Fusion, unweighted mean | −3.4%, CI spans zero — **not** reliably worse |
| Bone signal | +0.492 → **+0.098**, retracted |
| Reliability vs corruption | −0.813 (this one works) |
| Backbone reproduction | 45.149 mm vs 45.1 published (their own script gives 45.149) |
| Added trainable parameters | **0** |
| Canonicalization cost | 402 FLOPs/frame, 0.0005% of backbone |
| Audit / tests | 304 claims, 76 tests |

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
