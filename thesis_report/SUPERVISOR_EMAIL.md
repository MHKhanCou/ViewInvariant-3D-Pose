# Draft email to supervisor — send before he reads the report

Send this **before** the report, not with it. The two items below change how the
work is positioned, and he should not meet them for the first time in the PDF.

---

**Subject:** Thesis report — two positioning changes from the final literature review

Dear Sir,

The thesis report is attached. Before you read it I want to flag two changes I
made during the final literature review, because both affect how the
contribution is framed and I did not want them to come as a surprise.

**1. Our frame construction is a known algorithm.** The Gram-Schmidt body frame
we build from the torso and hip axes is the TRIAD algorithm, published by Black
in 1964 for determining spacecraft attitude from two direction measurements. The
accompanying rule — that the more accurately known direction should be the
primary axis — is due to Shuster and Oh (1981). Our axis-length finding, that a
frame should be built from the longest available segment, is that rule applied to
anatomy.

**2. The error propagation is also established, in biomechanics.** Our relation
that a direction read from two joints a distance L apart is uncertain by about
2σ/L restates results developed by Della Croce, Cappozzo and colleagues (1999,
2005) for anatomical landmark misplacement in motion capture. An earlier draft
presented it as our derivation. It is not, and the report now says so explicitly.

I have rewritten the positioning accordingly. The thesis no longer claims to have
invented the frame or discovered the principle. What it claims is:

- the **transfer** of that reasoning to joints inferred by a frozen network
  rather than markers placed by an examiner, and
- principally, the **experimental boundary** — three pre-registered tests
  establishing that axis length governs the choice *between* frame
  constructions, and governs neither which frame to trust within a construction
  nor which joint will disagree. Two of the three tests failed, and the failures
  are the result.

I also withdrew a second claim. The report previously argued that a fitted
constant agreeing to within one percent across two backbones showed the mechanism
was physical. I bootstrapped it and the confidence interval turned out wider than
the estimate, so the agreement carries no weight. That argument is withdrawn in
the text rather than quietly removed.

**3. A result demoted.** The per-limb multi-scale extension previously reported a
55.1 percent improvement. Checking it, I found that each limb frame is built from
exactly the three joints it is then scored on, which removes the orientation
being measured by construction rather than by the method working. A control
against a per-segment Procrustes floor confirms it: those levels sit within 13 to
23 percent of the best any rotation could do, where the global frame sits at 46
percent above its floor. I have demoted that figure to an exploratory measurement
and rewritten the surrounding claims. The main cross-view result, 74.1 percent
over 180 held-out pairs, is unaffected — its frame uses four constructor joints
out of seventeen scored, so thirteen are held out.

**On the title.** The report currently reads *"A Lightweight, Training-Free,
Reliability-Aware Geometric Canonicalization Framework for Cross-View
Comparability of Frozen Monocular 3D Pose Predictions."* I would like your
confirmation on it rather than assuming it stands, since I do not believe it has
been formally approved.

I think it is accurate as written, and deliberately so: it says *canonicalization
framework* rather than pose estimation, because the estimator is untouched and
only the coordinate frame changes; and *cross-view comparability* rather than
view-invariant estimation, because what improves is agreement between two
camera-relative predictions, not their accuracy. On "Reliability-Aware": the
component is falsified as an accuracy predictor five independent ways, but a
later experiment showed it does gate canonicalization quality on both backbones,
which is the function its own specification names. So the word is defensible for
that narrower claim and the report states which claim it is.

If you would prefer something shorter, or one that foregrounds the boundary
result, I am glad to change it — please let me know.

The report is 87 pages. Every reported number is recomputed from stored result
files by an automated audit (167 claims), and the pre-registrations for each
experiment are in the version history with timestamps preceding the results.

Thank you for your guidance throughout.

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
