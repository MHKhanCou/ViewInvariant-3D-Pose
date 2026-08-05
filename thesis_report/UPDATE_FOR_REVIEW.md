# Update for review — current state

Thesis: view-invariant 3D human pose estimation. Report due 6 Aug, defense 9 Aug.
This supersedes the previous version of this file, which was written before the
final correctness pass.

**Current state: 87 pages, 167/167 audit claims verified against stored
artifacts, 72/72 tests, abstract on one page, all 24 references cited.**

Since the last update the research was frozen and the work turned to correctness
and communication. **No new experiment was run.** Four things were withdrawn or
corrected, and every one of them made the thesis weaker on paper and stronger
under scrutiny.

---

## 1. A claim was withdrawn because it had no confidence interval

The report argued that a fitted constant agreeing across two independently
trained backbones — c = 21.3 and 21.5 mm, "within one percent" — showed the
mechanism was physical, and called this its strongest evidence.

That inference requires an interval on c, and none had been computed. A bootstrap
over the frame definitions gives:

| Backbone | c | 95% CI | width |
|---|---|---|---|
| MotionAGFormer-XS | 21.3 | **[19.1, 42.4]** | 23.4 mm |
| MotionBERT | 21.5 | **[19.2, 42.7]** | 23.5 mm |

The interval is wider than the estimate and the two nearly coincide, so two draws
landing one percent apart is unremarkable. **The argument is withdrawn** — in the
abstract, contributions, §5.16 and the conclusion — and the withdrawal is stated
rather than quietly dropped. What survives is the rank correlation, +0.904 and
+0.880, which was always the better-supported half.

**Worse than the claim was why it went unchecked.** The axis-law numbers had been
produced for two days by a scratch script that was never committed, so
`audit_numbers.py` verified none of them, and the σ stored in the artifact was
near zero from a stale division while the report quoted 7.5 from elsewhere. The
committed module now computes exactly what the report quotes, with eleven new
audit claims covering it.

## 2. The prior art was found, twice, and both times it narrowed the claim

**First: the construction is TRIAD.** The Gram-Schmidt body frame built from
torso and hip axes is the TRIAD algorithm (Black, 1964), the first practical
method for determining spacecraft attitude from two direction measurements. The
rule that the better-determined direction should be primary — which is the
axis-length principle — is Shuster and Oh (1981).

**Second, and this one cost more: the error propagation is established in
biomechanics.** An external reviewer flagged that "we derive the propagation from
joint noise to angular uncertainty" had become the heart of the thesis and should
be verified against biomechanics, motion capture and photogrammetry before
submission. It does not survive. Della Croce, Cappozzo & Kerrigan (1999)
propagated anatomical landmark calibration precision to bone geometry and joint
angles; Della Croce, Leardini, Chiari & Cappozzo (2005) assessed landmark
misplacement and its effect on joint kinematics. That literature states directly
that landmark imprecision propagates to anatomical frame orientation, and carries
the companion pivot relation `e = l·tan α` — which is, in substance, the Level-3
prediction §5.19 tests and finds to fail.

Equation (3.2) is therefore **a restatement, not a derivation**, and the report
now says so in §2.5, the abstract, the contributions and the conclusion.

**What survives, stated as three narrower things:**

1. **The transfer** — that reasoning was built for markers placed by a human
   examiner; we apply it where the landmarks are network outputs, whose error is
   dominated by coherent depth ambiguity rather than independent placement noise.
2. **The design consequence** — connecting the biomechanical propagation result
   to TRIAD's primary-axis rule for choosing a canonical axis.
3. **The empirical boundary — the actual contribution.** The rigid-segment pivot
   relation does not survive on an articulated skeleton. Neither aerospace nor
   single-segment biomechanics had reason to test that.

**The same literature also explains a failure the report previously could not.**
The SVD variant of the multi-landmark experiment is the *unweighted* solution to
Wahba's problem, whose optimality requires inverse-variance weights; over axes
spanning 138 to 461 mm those variances differ by an order of magnitude, and an
unweighted solution can fall below TRIAD on the best pair alone. The correctly
weighted variant then fails Wahba's *second* assumption, independence, which one
network predicting all seventeen joints violates. Both assumptions fail; the
two-vector construction survives because it needs neither.

## 3. The headline percentage did not reproduce from its own table

Found by trying to reject the report — recomputing its numbers from its tables.

§5.10 read *"reduces mean cross-view joint distance from 320.4 mm to 75.3 mm, an
improvement of 74.1 percent."* Those two numbers give **76.5 percent**. Every
improvement figure is the mean over camera pairs of each pair's own percentage,
while the sentence invites a ratio of aggregate means. The same gap appears on
the second backbone (77.6 against 81.0) and in the oracle-gap figures (90.5
against 91.1; 94.5 against 96.1).

Nothing was recomputed and no number changed — the per-pair convention is the
conservative one in every case. What changed is that the report now says so, in
the paragraph where a reader first reaches for a calculator, with six audit
claims pinning both conventions and asserting the quoted one is the smaller.

## 4. The README cited three papers wrongly

A correctness problem outside the .tex. The README contradicted the thesis's own
bibliography:

| Work | README said | Correct |
|---|---|---|
| VideoPose3D | *"…in the Wild by Adversarial Learning"*, **Joao Carreira** | *"…in Video with Temporal Convolutions"*, **Pavllo et al.** |
| MotionBERT | ICCV **2021**, "Wenjie Zhu, Moli Peng" | ICCV **2023**, Zhu, Ma, Liu, Liu, Wu, Wang |
| P-STMO | NeurIPS 2020, "Mikel Rodriguez" | ECCV 2022, Shan et al. |

Fabricated-looking author names in a public repository, with the BibTeX block
repeating one. It also still described the pre-thesis project — a baseline
reproduction with a demo — so anyone arriving from the report found different
work. Rewritten thesis-first, citations checked against the thesis bibliography.

---

## What the thesis now claims

The three-level boundary, each level pre-registered before running:

| Level | Question | Verdict |
|---|---|---|
| Between frame **constructions** | What should the frame be built from? | **Holds** — ρ = +0.904 / +0.880; acting on it raises multi-scale 25.6% → 55.1% |
| Between **frames** of one construction | Which frame to trust? | **Fails** — indistinguishable from a random null |
| Between **joints** of one frame | Which joint will disagree? | **Fails** — articulation dominates; torso-rigid joints sit at a *larger* radius yet disagree **2.5× less** |

Two of three failed, and the boundary is the contribution. Supporting results:
74.1% cross-view reduction over 180 held-out H36M pairs (179 improve, 90.5% of
the oracle gap), replicating at 77.5% on a second backbone with unmodified code;
the falsified reliability score gating canonicalization quality on both backbones
(reported as exploratory); and the bone-length retraction.

## Structural and presentational changes

- **Chapter 6 is now DISCUSSION, CONCLUSION AND FUTURE WORK**, with the
  Discussion as its first section — what transferred from rigid-body geometry,
  what did not and why, the two results that surprised us, three pieces of advice
  for anyone building on this, and a paragraph on why the work is worth reading
  despite being incremental.
- **Contribution list retagged by type**: [Analysis], [Negative result] ×2,
  [Validation], [Empirical finding], [Engineering], [Systems]. The axis-length
  principle is no longer listed as a contribution; it is inherited.
- **Narrative leads with the question**, not the framework — canonicalization is
  occupied prior work (3DPCNet, ICASSP 2026), the scope question is not.
- **`presentation/render.py`** generates the teaser, four report figures and the
  two-view comparison from stored artifacts, so a figure cannot drift from the
  number it illustrates. Three deliberate choices: the two-view picks the
  **sequence-median frame, not the most flattering**; panels share a scale so the
  canonical column cannot look tighter by being drawn smaller; and the level-2
  panel plots a reference curve purely to fix an honest vertical scale, since on
  a 2 mm axis a null result looks dramatic.
- **`FREEZE_CHECKLIST.md`** and **`SUPERVISOR_EMAIL.md`** added; `DEFENSE_QA.md`
  gains 10s/30s/2min/5min explanations and five new questions.

## Where the work is weakest — unchanged, and stated in the report

1. **The metric measures agreement, not correctness.** Two predictions wrong in
   the same way agree perfectly. The oracle floor and the retained correlation
   with ground-truth error bound this but do not eliminate it, and that check
   exists on one dataset only. **Still the sharpest attack.**
2. **No downstream task succeeds.** Retrieval is negative and used a superseded
   protocol. The honest ceiling on significance.
3. **Two datasets, both lab rigs; two backbones, both transformers.**
4. **The triage finding is exploratory** — a comparator in its pre-registration,
   not its subject.

## Questions worth putting to a reviewer

1. Is the boundary result — two pre-registered failures plus one confirmation —
   defensible as the *primary* contribution, or does an examiner read three
   experiments where two failed as a weak chapter?
2. Having found the prior art late and narrowed the claim twice in one week, is
   the reporting of those withdrawals a strength, or does volunteering them
   invite doubt about what else went unchecked?
3. Is anything still overstated after the TRIAD and Cappozzo corrections?
