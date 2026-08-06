# Draft email to supervisor — send before he reads the report

Send this **before** the report, not with it. Item 1 is a result that goes
against the project, and he must not meet it for the first time in the PDF or at
the defence.

---

**Subject:** Thesis report — a result against our own method, and three positioning changes

Dear Sir,

The thesis report is attached. Before you read it I want to flag four things,
because one of them is a finding that works against the method and the others
change how the contribution is framed. None of them should come as a surprise in
the PDF.

**1. A simpler baseline beats our method, and the report says so in the
abstract.** I ran the comparison a reader would ask for and had not been run:
aligning each predicted pose to a single fixed reference skeleton by the Kabsch
algorithm. It is training-free, label-free, calibration-free and single-view, so
it meets every requirement our framework claims as its profile. It beats our
anatomical frame on **all 180 held-out pairs of Human3.6M, under both
backbones**, and on **all fifteen actions**. The criterion and all three possible
readings were written down and committed before the experiment ran, so this is
the outcome the pre-registration committed me to reporting.

I have not hidden it or softened it. It is in the abstract, the contributions
list, Section 5.10, the Limitations section, and the opening of the conclusion.

What I would say if asked to defend the work in spite of it: the baseline wins on
the metric, but it **cannot run the experiment this thesis is about**. Kabsch
alignment has no anatomical axis, so there is no axis to hold fixed and vary, and
therefore no way to ask what governs whether a body frame is consistent across
viewpoints. The contribution is that boundary, and the instrument that makes it
askable — not the number. I would rather present it that way than argue the
baseline away.

**2. Our frame construction is a known algorithm.** The Gram-Schmidt body frame
built from the torso and hip axes is the TRIAD algorithm, published by Black in
1964 for spacecraft attitude determination. The accompanying rule — that the more
accurately known direction should be primary — is Shuster and Oh (1981). Our
axis-length finding is that rule applied to anatomy.

**3. The error propagation is also established, in biomechanics.** The relation
that a direction read from two joints a distance L apart is uncertain by about
2σ/L restates results developed by Della Croce, Cappozzo and colleagues (1999,
2005) for anatomical landmark misplacement. An earlier draft presented it as our
derivation. It is not, and the report now says so explicitly.

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
  flatters us. It is 72.2 percent over 180 held-out pairs with 179 improving; the
  seventeen-joint figure, 74.1 percent, is given alongside with the reason.

I also found and fixed an error in two of my own figures while preparing the
final version: they were plotting all 85 records in a results file rather than
the 29 the section reports on, which made the plotted mean disagree with the
figure quoted in the text. The text was correct throughout; the figures were not.
Both are regenerated and the generator now asserts the record count.

**On the title.** The report reads *"A Lightweight, Training-Free,
Reliability-Aware Geometric Canonicalization Framework for Cross-View
Comparability of Frozen Monocular 3D Pose Predictions."* I would like your
confirmation rather than assuming it stands, as I do not believe it was formally
approved. It says *canonicalization framework* rather than pose estimation
because the estimator is untouched, and *cross-view comparability* rather than
view-invariant estimation because what improves is agreement between two
camera-relative predictions, not their accuracy.

**One word in it I am not certain of, and would rather raise than defend
later: "Reliability-Aware".** The report falsifies the reliability score as an
accuracy predictor along five independent axes, and the pre-registered
conditioning criterion fails. What survives is narrower: the score does gate
canonicalization *quality* on both backbones, which is the function its own
specification names. So the word is defensible for that narrower claim and the
report states plainly which claim it is — but it sits in the most prominent
position in the document and describes a component the body largely falsifies.
If you think it overstates, dropping it to *"A Lightweight, Training-Free
Geometric Canonicalization Framework for Cross-View Comparability of Frozen
Monocular 3D Pose Predictions"* costs nothing and I would prefer that to
defending it at the viva. Your call, and I am glad either way.

The report is 90 pages. Nine experiments were pre-registered with their criteria
committed to version history before each run; **five failed their own criteria
and a sixth returned a competing method as the better one**. Every reported
number is recomputed from stored result files by an automated audit of 248
claims, alongside 76 unit tests, and both pass.

Thank you for your guidance throughout — particularly the instruction to justify
every claim against evidence, which is what produced the negative results above
rather than a report that only reported what worked.

Best regards,
Mehedi Hasan Khan
ID 12108004
Session 2020–21, Department of Computer Science and Engineering

---

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
