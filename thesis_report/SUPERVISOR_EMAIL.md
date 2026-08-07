# Draft email to supervisor — send before he reads the report

Send this **before** the report, not with it. Item 1 is a result that goes
against the project, and he must not meet it for the first time in the PDF or at
the defence.

---

**Subject:** Thesis report — how the work developed, and a result against my own method

Dear Sir,

The thesis report is attached, along with a one-page walkthrough
(`WORKFLOW.md`) that answers the two questions you asked me: what the system is
actually doing, and how the output compares to base MotionAGFormer. Before you
read either, let me say briefly how the work got from your proposal to this
report.

**How it developed.** Your proposal set the direction: the same person seen from
different cameras produces different 3D coordinates, because each prediction is
expressed in that camera's own frame, which makes cross-view comparison
impossible. I studied the literature and ran the `pose-estimation-3d`
repository you shared, and the output I showed you came from it. From there I
decided to keep the estimator **completely frozen** rather than train a new
network. I selected **MotionAGFormer** myself as the lifting backbone, and later
added **MotionBERT** to check that the observations were not an artifact of one
model. The pipeline became:

> RGB video → 2D detection → MotionAGFormer (frozen) → training-free body-frame
> canonicalization → cross-view comparable 3D pose

The idea is to leave the estimator untouched and transform its predicted
skeleton into a body-centred coordinate system built from anatomical landmarks,
so predictions from different viewpoints become comparable without retraining
and without calibrating the cameras.

**Your two questions, directly.** The estimator's accuracy does not improve, and
cannot: it is frozen and canonicalization adds zero parameters. Action-balanced
MPJPE on Human3.6M is **45.149 mm before and after**, matching the backbone's own
published figure. What improves is agreement between cameras: mean cross-view
joint distance falls from **372.7 mm to 93.4 mm**, a 72.2 percent reduction over
180 held-out camera pairs, with 179 of 180 improving and 75.8 percent on the
second backbone. Those are the same predictions, expressed in a different frame.

The early results were good, but I did not stop there. Instead of asking only
whether the method works, I investigated **why** it works and **what determines
the quality of a body-fixed frame**. That became the research question of the
thesis. Along the way I found that several ideas I believed were mine were
already established, and I have said so rather than claim them.

Four things I would rather you hear from me than meet for the first time in the
PDF. One of them works against the method; the others change how the
contribution is framed.

**1. A simpler baseline beats my method, and the report says so in the
abstract.** I ran the comparison a reader would ask for and that had not been
run: aligning each predicted pose to a single fixed reference skeleton by the
Kabsch algorithm. It is training-free, label-free, calibration-free and
single-view, so it meets every requirement my framework claims as its profile.
It scores **57.5 mm against my 93.4 mm** and beats the anatomical frame on **all
180 held-out pairs of Human3.6M, under both backbones**, and on **all fifteen
actions**. The criterion and all three possible readings were written down and
committed before the experiment ran, so this is the outcome the pre-registration
committed me to reporting.

I have not hidden it or softened it. It is in the abstract, the contributions
list, the section "A Single-View Baseline, and It Wins", the Limitations section, and the opening of the conclusion.

What I would say if asked to defend the work in spite of it: the baseline wins on
the metric, but it **cannot run the experiment this thesis is about**. Kabsch
alignment has no anatomical axis, so there is no axis to hold fixed and vary, and
therefore no way to ask what governs whether a body frame is consistent across
viewpoints. The contribution is that boundary, and the instrument that makes it
askable — not the number. I would rather present it that way than argue the
baseline away.

**2. My frame construction is a known algorithm.** The Gram-Schmidt body frame
built from the torso and hip axes is the TRIAD algorithm, published by Black in
1964 for spacecraft attitude determination. The accompanying rule — that the more
accurately known direction should be primary — is Shuster and Oh (1981). My
axis-length finding is that rule applied to anatomy.

**3. The error propagation is also established, in biomechanics.** The relation
that a direction read from two joints a distance L apart is uncertain by about
2σ/L restates results developed by Della Croce, Cappozzo and colleagues (1999,
2005) for anatomical landmark misplacement. An earlier draft presented it as my
own derivation. It is not, and the report now says so explicitly.

The thesis therefore no longer claims to have invented the frame or discovered
the principle. It claims the **transfer** of that reasoning to joints inferred by
a frozen network rather than markers placed by an examiner, and principally the
**experimental boundary**: axis length governs the choice *between* frame
constructions, and governs neither which frame to trust within a construction nor
which joint will disagree, because articulation dominates there.

**4. Two results corrected downward, and one headline changed.**

- The per-limb multi-scale extension previously reported 55.1 percent. Each limb
  frame is built from exactly the three joints it is then scored on, which
  removes the orientation being measured by construction rather than by the
  method working. A per-segment Procrustes control confirms it. Demoted to an
  exploratory measurement.
- The report previously argued that a fitted constant agreeing to within one
  percent across two backbones showed the mechanism was physical. The bootstrap
  interval is wider than the estimate, so it carries no weight. Withdrawn in the
  text rather than quietly removed.
- The headline cross-view figure now **excludes the four joints the frame is
  built from**, since the construction pins them and any seventeen-joint average
  flatters me. It is 72.2 percent over 180 held-out pairs with 179 improving; the
  seventeen-joint figure, 74.1 percent, is given alongside with the reason.

I also found and fixed an error in two of my own figures while preparing the
final version: they were plotting all 85 records in a results file rather than
the 29 the section reports on, which made the plotted mean disagree with the
figure quoted in the text. The text was correct throughout; the figures were not.
Both are regenerated and the generator now asserts the record count.

**On the title.** The report reads *"A Lightweight, Training-Free Geometric
Canonicalization Framework for Cross-View Comparability of Frozen Monocular 3D
Pose Predictions."* I would like your confirmation rather than assuming it
stands, as I do not believe it was formally approved. It says *canonicalization
framework* rather than pose estimation because the estimator is untouched, and
*cross-view comparability* rather than view-invariant estimation because what
improves is agreement between two camera-relative predictions, not their
accuracy.

**One change I made to it myself, and would rather report than be asked about.**
An earlier draft included the word *"Reliability-Aware"*. I have removed it. The
report falsifies the reliability score as an accuracy predictor along five
independent axes, and the pre-registered conditioning criterion fails. What
survives is narrower: the score does gate canonicalization *quality* on both
backbones, which is the function its own specification names. That narrower
finding belongs in Chapter 5, where it is reported with its controls — but it did
not belong in the most prominent line of the document, describing a component the
body largely disproves. The shorter title claims only what the evidence supports.

The report is 55 pages, of which 33 are the thesis proper and the rest appendices and references, with a 97-page technical version attached as a supplement in case you want the full evidence. Seventeen experiments were pre-registered with their criteria
committed to version history before each run; **more than half failed their own
criteria, and one returned a competing method as the better one**. Every
reported number is recomputed from stored result files by an automated audit
of 304 claims, alongside 76 unit tests, and both pass. A separate script
verifies that every pre-registration was committed before its own result, and
it passes on all sixteen documents.

Throughout, I tried to put scientific correctness ahead of preserving my earlier
hypotheses. Whenever an experiment or the literature contradicted an assumption,
I revised the report rather than the evidence, so what remains is only what both
support.

I would value your comments on the final report before the defence, particularly
on its positioning, its contribution and its title.

Thank you for setting the direction in your proposal document and for the
repository you shared — the gap it named, that existing methods lack explicit
geometric priors and view-invariant constraints, is unchanged and is the gap
this report addresses.

Best regards,
Mehedi Hasan Khan
ID 12108004
Session 2020–21, Department of Computer Science and Engineering

---

**Attach:** `Thesis_12108004.pdf` (the 55-page submission), `WORKFLOW.md`, and
`fig_realview.png`. Optionally `Thesis_12108004_supplement.pdf`, the 97-page
technical version, if you want him to have the complete evidence.

**Add one question before sending:** whether the department sets a minimum page
count for the report. Fifty-five clears most rules, but it is his to confirm
and it costs one sentence to ask.

---

## If he asks "what is your model actually doing?"

> Sir, I do not generate a new pose. The exact same MotionAGFormer prediction is
> re-expressed in a coordinate system built from the person's own hips and torso.
> Then I measure how closely two cameras agree. Mean cross-view joint distance
> drops from 372.7 mm to 93.4 mm across 180 held-out pairs.

## If he asks "how much better is it than base MotionAGFormer?"

> Its accuracy is not better, Sir — it is identical, 45.149 millimetres either
> way, because the estimator is frozen and I add zero parameters. What improves
> is cross-view consistency: two cameras that disagreed by 372.7 mm now agree to
> 93.4 mm. And a simpler Kabsch baseline reaches 57.5 mm, so on that metric it
> beats mine; the report says so in the abstract.

Never call the cross-view number MPJPE. MPJPE is error against ground truth and
it does not move. This is *cross-view joint distance*, prediction against
prediction.

## If he asks "so what is actually yours?"

Give this, in one breath:

> The boundary. Everyone knew the frame construction and everyone knew the error
> propagation. Nobody had tested how far that reasoning carries on an articulated
> body reconstructed by a network, and the answer is: less far than it looks. It
> decides what to build the frame from, and nothing finer. Joints rigidly
> attached to the torso are further from the root than the knees and yet agree
> two and a half times better across views, which contradicts the rigid-body
> prediction in the wrong direction for it to be a matter of degree.

## If he asks "then why keep the method at all, if Kabsch wins?"

> Because Kabsch cannot run the experiment. It has no anatomical axis, so there
> is no variable to hold fixed and vary, and the question of what governs frame
> consistency cannot be posed inside it. The anatomical frame is the instrument
> that makes the boundary measurable. The report says plainly that as a way of
> reducing cross-view distance, the simpler method is better on this data.

## If he asks "why report something that damages your own result?"

> Because the criterion was pre-registered and committed before the run. If I
> only reported the pre-registrations that came out well, none of the others
> would mean anything either.
