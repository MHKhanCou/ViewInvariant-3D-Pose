# External Review: Adversarial Critique of Full_Thesis_Report.tex

Prepared 2026-08-06 in the role defined in `EXTERNAL_REVIEW_PROMPT.md`. Independent read.
The thesis's own admitted findings (items 1-8 of the prompt) are not re-reported as
findings here. Every claim below was checked against the report text, the nine
pre-registrations, and the evaluation code that produces the audited artifacts.

Section numbers below refer to the .tex source; the defence happens in three days,
so each finding ends with either a three-day fix or "no fix — prepare an answer".

---

## Findings, ranked by severity (worst first)

### FINDING 1 (severity: HIGH — this is the circularity you asked me to hunt, and it is still in the report)

**Claim.** The circularity control of §5.16 is complete, and its conclusion is stated
as: "The circularity identified here is a property of three-joint limb segments, not
of the method."

**Where.** Table 5.9 (the headroom table) and the two paragraphs after it in §5.16.

**Why it is weak.** The headroom table lists the global frame (1.46×), the shipped
arms (2.39×/2.55×) and the long-axis limbs (1.13×–1.23×). It **omits the torso level**.
I checked `thesis_artifacts/multiscale_control/control.json` (the very artifact the
table claims to summarise). The torso level is present in the artifact:

| set | segment | n_scored | n_builders | canonical mm | oracle mm | headroom |
|---|---|---|---|---|---|---|
| all three | torso | 6 | **4** | 29.08 | 23.92 | **1.216×** |

The torso segment is defined as ids `[7,8,9,10,11,14]` with y from `(7,10)` and x from
`(11,14)` (`canonical/multiscale.py:31`). Four of its six scored joints are constructor
joints. Its headroom of 1.216× sits **inside the exact band (1.13×–1.23×) the report
uses to call the long-axis limbs "substantially mechanical"** — and the torso row is
silently absent from the published table. The sentence "not of the method" is therefore
false as written: the *torso level of the same method* has the same defect, at the same
magnitude.

**Consequence.** The multi-scale *combined* headline (46.2→28.2 mm, +25.6→+55.1%) is
NOT directly corrupted, because `combined_from_ms` (evaluation/h36m_multiscale.py:128)
scores TRUNK `[0,7,8,9,10]` in the *global* frame and only the four limb segments in
their own frames — the torso segment frame never enters the combined metric. So the
headline survives. What is corrupted is the *argument*, in two places:

1. §5.13.2: "The corrected levels support the diagnosis independently… All five now lie
   between 26.3 mm and 30.7 mm… That convergence was not optimized for… and it is the
   strongest evidence that axis length, not anatomy, was driving the spread." The torso
   (28.6 mm) is one of the five "converged" levels, and its low value is 4/6 mechanical.
2. The demotion's own framing: if 1.13–1.23× headroom means "almost nothing remained for
   the frame to remove", that verdict applies verbatim to the torso at 1.216×.

**Three-day fix.** Add the torso row to Table 5.9 (it is already in the artifact — the
fix is one table row and one sentence). Reword §5.13.2's "strongest evidence" sentence
to attribute the convergence to the global-frame comparison only, and add one clause to
§5.16: "the torso level of the multi-scale construction is built from four of its six
scored joints and sits at 1.22× its own floor, so the limb-level demotion applies to it
as well; the combined metric is unaffected because the trunk is scored in the global
frame." A prepared answer must be ready either way, because the artifact makes this
findable in minutes.

---

### FINDING 2 (severity: HIGH — a number you will be asked about that the report cannot currently explain)

**Claim.** The global frame's canonical cross-view distance is reported as **75.3 mm**
(Table 5.12, abstract, §5.10) and as **76.2 mm** in Table 5.9 (explained there as the
pooling convention), and as **62.7 mm** in the multi-scale section (§5.13, Table 5.8:
"Global frame 62.7 mm" and `mean_global_distance_mm` = 62.714 in
`thesis_artifacts/h36m_multiscale/h36m_multiscale.json`).

**Where.** §5.10 vs §5.13 vs Table 5.9.

**Why it is weak.** The 62.7 vs 75.3 gap is 17%, and it is *not* the pooling convention:
the two experiments use different evaluation paths. The cross-view headline canonicalizes
via `canonicalize_stream` (temporal hip-sign flip, 27-frame windows, per-pair means); the
multi-scale "global" baseline canonicalizes each frame with `canonicalize_with`
(per-frame, no temporal flip) and averages over frame-pairs pooled inside each camera
pair. The report says the multi-scale comparison "take[s] both quantities from the same
code path so that the baseline and the treatment differ only in the frame used" — true
*within* the multi-scale module — but it never reconciles 62.7 with the 75.3 headline.
An examiner dividing Table 5.8 by Table 5.12 will ask which number is the canonical
distance of the global frame, and the report currently has no stated answer.

**Three-day fix.** One paragraph in §5.13: state that the multi-scale baseline (62.7 mm)
is a per-frame recomputation under the multi-scale code path and is not the same quantity
as the streamed headline (75.3 mm), exactly as §5.10 already reconciles 75.3 vs 76.2.
Better: rerun the shipped multi-scale evaluation with the streamed global path, or at
least recompute the +25.6% against 75.3 in one line and report both. Do not leave the
discrepancy implicit.

---

### FINDING 3 (severity: MEDIUM-HIGH — the defence rehearsal material contradicts the frozen report)

**Claim.** The defence preparation (`thesis_report/DEFENSE_QA.md`) is the document you
will rehearse from, and it is stale relative to the frozen report in exactly the places
an examiner probes.

**Where.** DEFENSE_QA.md throughout; compare with the report, UPDATE_FOR_REVIEW.md and
the artifacts.

**Why it is weak.** Specific contradictions:

1. **The 2-minute answer still leads the multi-scale story with +55.1% "at no cost"** and
   the "rank correlation 0.90 and 0.88" as the axis-length evidence. Both are demoted in
   the report: §5.16 says the 55.1% rise "can no longer be presented as evidence that
   longer axes produce better frames" and the rank correlations are "confounded with
   constructor count." Volunteering either in the viva will look like you have not
   internalised your own retraction — the single most damaging possible impression.
2. **The "Numbers to Know Cold" table quotes 74.1%** (all-17) as the headline; the
   abstract now leads with **72.2%** on the thirteen non-constructor joints. Both are in
   the report, so this is minor, but the 30-second pitch says "74 percent" while the
   abstract says 72.2 — say the one that matches the abstract.
3. **Claim counts disagree across documents**: DEFENSE_QA header says "131 claims",
   its 5-minute script says "167 claims", UPDATE_FOR_REVIEW says "180 claims / 72 tests",
   the report says "248 numerical claims… 76 tests" (§4.7). The report itself concedes
   these self-referential numbers "carry only our care rather than a guarantee" — and
   they are currently *wrong in three documents at once*. An examiner who asks "how many
   claims does your audit check?" will get a different answer depending on which document
   you quote.
4. DEFENSE_QA says "four pre-registrations" (5-minute script) where the report says nine.

**Three-day fix.** Rewrite DEFENSE_QA.md to match the frozen report: 72.2% headline,
55.1% demoted (with the one-line reason), 248 claims / 76 tests / nine pre-registrations
(or whatever single pair of numbers you settle on), and the torso caveat from Finding 1.
This is a one-hour mechanical job and it removes the highest-probability self-inflicted
wound on defence day.

---

### FINDING 4 (severity: MEDIUM — a surviving overclaim, in the title of all places)

**Claim.** The title: "A LIGHTWEIGHT, **TRAINING-FREE**, **RELIABILITY-AWARE** GEOMETRIC
CANONICALIZATION FRAMEWORK…" The report's own Limitations state the reliability score
was "falsified as an accuracy predictor along five independent axes", and §5.15 confirms
the pre-registered conditioning criterion "fails". The reliability score works only as a
canonicalization-quality gate (§5.16.1), which the report itself labels exploratory.

**Where.** Title page; DEFENSE_QA.md Q4 defends it.

**Why it is weak.** "Reliability-Aware" is the strongest surviving overclaim because it
sits in the document's most prominent position and describes a component the body of the
report falsifies. The defence prepared for this (Q4) and the answer is honest — but the
answer depends on the examiner granting that "gate, not predictor" is what "Reliability-
Aware" means. It is a genuine overclaim only in the title; the abstract and body are
accurate.

**Three-day fix.** You said the report is frozen and the title was never supervisor-
approved (UPDATE_FOR_REVIEW.md: "The title was never supervisor-approved"). Ask the
supervisor to approve either the current title (with the prepared Q4 answer) or a
retitle to "Geometric Canonicalization…" (dropping "Reliability-Aware"). One email, one
edit. If you cannot change it, do not volunteer the title in the opening; let the
abstract define the claim.

---

### FINDING 5 (severity: MEDIUM — statistical handling is mostly right; one asymmetry is exposed)

**Claim.** The report says the per-pair mean convention "is the lower of the two in every
case we report, so the choice is conservative" (§5.10), and discloses both conventions
everywhere including the fusion section.

**Where.** §5.10 (cross-view), §5.17 (fusion).

**Why it is weak.** The statement "conservative in every case we report" is true for the
cross-view improvement (72.2 vs 76.5) and the oracle gap (90.5 vs 91.1), but **false for
the fusion headline**: the median-fusion headline +8.4% is the *ratio of aggregate means*
(the favourable convention, whose interval excludes zero), while the per-frame mean
improvement is +4.7% with an interval that spans zero (§5.17). The report discloses this
asymmetry explicitly and honestly ("reporting only the first would select the convention
that favours the claim, having selected the conservative one elsewhere") — so this is
not a hidden error. It is, however, a target: an examiner can fairly say you used the
favourable convention where it favoured you and the conservative one where it did not.
The bootstrap machinery itself (cluster bootstrap, 10,000 draws, 30 groups, seed 12345;
`evaluation/h36m_crossview.py:140`) is correct and properly matches the stated unit of
dependence; the 2σ/L derivation, the c = 2√2σ constant, and the radial-law slope band
[0.038, 0.073] all re-derive correctly from the code.

**Three-day fix.** No code fix. Prepare the answer: "the per-pair convention is the
honest one for improvements defined per pair; the fusion experiment is defined on pooled
frames, so the ratio of aggregate means is the natural unit there, and we report both
rather than choosing for the reader." That is the strongest truthful version, and it is
already almost word-for-word in the report.

---

### FINDING 6 (severity: LOW-MEDIUM — an overclaim that survives in the contributions list)

**Claim.** §1.4 Contributions, [Analysis]: "Axis length governs the choice between frame
constructions… the longer shoulder axis wins on both backbones, by 5.2 and 4.4 percent,
with intervals excluding zero."

**Where.** §1.4, §5.16, conclusion.

**Why it is weak.** This is the single surviving positive claim of the whole boundary
argument, and it rests on **one paired comparison** (hip axis vs shoulder axis) on two
backbones. The effect (71.4 vs 75.3 mm; 57.5 vs 60.1 mm) is real but small — 5.2% on a
metric that is agreement, not accuracy — and its confidence intervals are wide
([+1.6, +7.8] and [+1.7, +3.6]). The report is admirably precise about *not* citing the
limb-level correlations (correctly demoted), which makes this single comparison load-
bearing. The wording "governs the choice" is stronger than "is associated with a small
improvement in one paired comparison on two backbones." An examiner can concede
everything and still say: one comparison, two backbones, both transformers, both trained
on the same benchmark family.

**Three-day fix.** No code fix (report frozen). Prepare: (a) the replication on two
unrelated backbones is the strength — 19× size difference, no shared architecture,
byte-identical code; (b) the effect is small but the direction and the intervals were
pre-registered before the number existed; (c) if asked "is 5.2% worth a contribution",
answer "the contribution is the boundary, not the effect size" — and see the defence
argument below, because this is the same line.

---

### FINDING 7 (severity: LOW — an internal inconsistency a careful reader will find)

**Claim.** §4.5 says "the automated audit re-derives 248 headline claims"; §5.2 says
"an automated audit covering forty claims." §1.4 says "248 numerical claims"; the
abstract-era UPDATE_FOR_REVIEW says "180 claims"; DEFENSE_QA says 131/167.

**Where.** §4.5, §5.2, §1.4, and the support documents.

**Why it is weak.** The report itself says its self-referential counts are not audited
("It is the one category of number here that carries only our care rather than a
guarantee, and two were found stale during the final week"). They are, in fact, stale
*again*: the report quotes 248 and 76 tests in one place and 40 claims in another. This
is exactly the class of number an examiner will pick because it is cheap to check.

**Three-day fix.** Grep the report and all support documents for every occurrence of
"claims", "tests", "pre-registrations" and make them one pair of numbers each. 20
minutes. Then add one line to the audit's self-check block (it already has the mechanism
for this).

---

## The seven questions you asked, in order

### 1. Correctness (bootstrap, averaging conventions)

Handled correctly, with one exposed asymmetry (Finding 5). The cluster bootstrap resamples
whole subject-action groups (30 clusters), 10,000 draws, fixed seed, and the interval is
computed on the per-pair statistic, matching the stated unit of dependence — verified in
`evaluation/h36m_crossview.py:140-167`. The disclosure of the two averaging conventions
is adequate in the cross-view and oracle-gap sections and exemplary in the fusion section
(where the two conventions *disagree about significance*, and the report says so in so
many words). The derivation chain 2σ/L → d² ≈ (c·r̄/L)² + d₀² with c = 2√2σ → radial
slope band [0.038, 0.073] re-derives correctly from the code and the stated constants
(σ = 7.5 mm from the limb fit; L_hip = 275.8; L_torso = 456.1). No statistical error found
in the machinery itself. The only statistical-adjacent issues are the fusion convention
asymmetry (Finding 5) and the pooling-vs-stream discrepancy (Finding 2).

### 2. Circularity — is there another one?

**Yes: the torso level.** See Finding 1. It is the same defect (scored on joints the frame
is built from) at the same magnitude (1.216× headroom, inside the demotion band), and it
is omitted from the table that claims to report the control. The headline multi-scale
number survives because the combined metric scores the trunk in the global frame — but
the report's own sentence "a property of three-joint limb segments, not of the method" is
not true, and the "strongest evidence" convergence argument in §5.13.2 leans on a
partly-mechanical torso row. No other fit/score overlap was found in the remaining
numbers: the global-frame headline (13 non-constructor joints, 1.46× floor) is clean; the
template fit sets are disjoint from scored sets in the ablation that matters (§5.14.2);
the oracle is a floor, not a competitor; and the triage test scores the reliability score
against a target it does not overlap.

### 3. Surviving overclaims

- **The title's "Reliability-Aware"** (Finding 4) — the strongest survivor.
- **§5.13.2's "strongest evidence that axis length, not anatomy, was driving the spread"**
  — survives only because Finding 1 was missed; once the torso is in the table, this
  sentence is a retracted-adjacent claim still standing in the body.
- **§1.4 [Analysis] "governs the choice"** (Finding 6) — overstated by one degree; the
  evidence is one small paired comparison.
- **§4.5's "248 headline claims"** alongside §5.2's "forty claims" (Finding 7) — an
  inconsistency, not an overclaim, but it reads as one.
- Everything else the prompt lists (1-8) is already conceded in the report and is not
  re-found here. The abstract's "closing 87.0 and 91.4 percent of the gap to a per-frame
  Procrustes oracle" is fine as stated (it is the 13-joint recomputation, §5.10).

### 4. Underclaims

Two places where the report under-sells what it has:

- **The triage result is stronger than "exploratory".** §5.16.1 shows the reliability
  score, falsified five ways as an *accuracy* predictor, monotonically gates
  *canonicalization quality* on two backbones with intervals excluding zero and a
  confound control. The report labels it exploratory because it was a comparator within
  the conditioning pre-registration. That is honest, but the asymmetry with the
  bone-length signal (which got a full section, a retraction, and a mechanism) is
  striking: the triage result is *more* replicated than the bone-length signal ever was
  before it was retracted. It could be stated as "the score's five falsifications are
  bounds on what it predicts, not a verdict that it predicts nothing" — which the report
  does say — but the finding could carry one more line of confidence without
  overclaiming.
- **The cross-view distance → GT-error correlation (+0.601 vs +0.188 raw, §5.4)** is a
  genuinely useful deployed signal (label-free, calibration-free, two cameras) and is
  buried mid-chapter as an "anchoring" check. It is arguably the strongest *usable*
  contribution after the template result, and it is the one thing that answers "so what?"
  partly. The report could present it as a standalone result rather than a sanity check.
  (Only "partly" — it is still a correlation on one dataset.)

### 5. Prior art I have missed

Checked against the report's bibliography and a fresh web sweep of 2024-2026:

- **3DPCNet (arXiv:2509.23455, 2025)** exists and is the closest competitor — a learned,
  estimator-agnostic canonicalizer whose own paper reports a hand-built anatomical
  baseline of the same family as yours losing to it. Cited correctly. This is the one the
  examiner will know.
- **MoViD (arXiv:2604.03299, 2026)** exists; **V-VIPE (CVPR-W 2024, arXiv:2407.07092)**
  exists; **Pr-VIPE (ECCV 2020)** exists. All three are cited and positioned correctly.
- **No 2024-2026 paper doing training-free anatomical body-frame canonicalization as
  post-processing** was found. The gap you claim (the requirement profile) appears to be
  genuinely unoccupied in the recent literature. The closest analogues are geometric
  baselines inside learned-paper evaluations (3DPCNet's own), which is exactly what your
  report says.
- **The one literature family worth one more line**: biomechanics *gait-analysis*
  canonicalization (anatomical-frame definitions for joint angles, e.g. Cappozzo et al.
  1995, and the ISB recommendations Wu & Cavanagh 1995). You cite Della Croce 1999/2005,
  which are the right sources for the error-propagation claim, but an examiner from
  biomechanics/motion capture may ask whether "build an anatomical frame from hip and
  torso landmarks" is standard gait-analysis practice. It is, for *marker-based* data;
  your transfer to *network-inferred joints* is the defensible part, and that is exactly
  the line the report already draws. One sentence acknowledging the gait-analysis usage
  would pre-empt the question; it is not currently there.
- **Kabsch-to-template as a pose-normalisation baseline** exists in the action-recognition
  literature (e.g., "Normalized Human Pose Features for Human Action Video Alignment",
  ICCV 2021, uses Kabsch alignment to a reference skeleton). Your template experiment is
  therefore not novel *as a baseline* — but you never claim it is, and you cite V-VIPE's
  use of the same preprocessing. If an examiner names a specific paper, the correct
  response is "yes, and that is exactly why the baseline was the obvious comparator;
  the report pre-registered it against itself."

### 6. Examiner simulation — five hardest questions, in order

1. **"You have shown that a simpler method beats yours on every pair, every action, every
   centring, under three templates. Why does the anatomical frame exist?"** — This is the
   question the whole defence rests on; you have a prepared answer (§10c in DEFENSE_QA).
   It is sound but it must be delivered without a trace of defensiveness. **Difficulty for
   you: medium — you have rehearsed it.**
2. **"Your own abstract admits the reliability score in your title was falsified five
   ways. Is 'Reliability-Aware' honest?"** — Q4 in DEFENSE_QA. **Difficulty: medium-low —
   you have an answer; the risk is only that you sound defensive.**
3. **"Your circularity table omits the torso level, which your own artifact shows at
   1.22× its floor with four of six scored joints building the frame — the same band you
   used to demote the limbs. What survives of the multi-scale argument?"** — This is the
   question you are NOT prepared for (Finding 1). **Difficulty: HIGH. You would struggle
   to answer from the report alone, because the report does not contain the answer.**
   The prepared answer is short: the combined metric scores the trunk in the global
   frame, so the +25.6→+55.1% combined numbers stand; what changes is the per-level
   interpretation and the convergence sentence.
4. **"Why is the global frame 75.3 mm in one table and 62.7 mm in the multi-scale table?
   Which is the canonical distance?"** — Finding 2. **Difficulty: HIGH, for the same
   reason — the report does not currently answer it.** One prepared paragraph fixes it.
5. **"One clean comparison, 5.2% and 4.4% with wide intervals, on two backbones that are
   both transformers trained on Human3.6M. Is that enough to carry the boundary as your
   contribution?"** — Finding 6. **Difficulty: medium — the honest answer is "the
   contribution is the boundary, not the effect size", and the report gives you the
   language for it.**

Order rationale: they will open with the result that is in your own abstract (Q1), move
to the title (Q2), then probe the one control they can find a hole in (Q3), then a number
inconsistency (Q4), and finish with the contribution-size question (Q5). Q3 and Q4 are
the two you would struggle with; both have prepared answers above.

### 7. Is the contribution sufficient for an undergraduate thesis?

Honest answer: **yes, but only because of how the boundary is framed, and the frame is
the fragile part.** The claimed contribution — "the experimental boundary" — is a
negative result dressed up, and the report itself says so more honestly than most theses
ever do ("We regard the bound as more useful than a third confirmation would have been",
"five of the nine pre-registered experiments failed their own criteria"). For an
undergraduate thesis, that is defensible *and* rare: the pre-registration discipline, the
audit trail, the two-dataset/two-backbone replication and the systematic retraction of
your own claims constitute real methodological work that most examiners will credit
heavily. What would *not* be defensible is presenting the boundary as a discovery; the
report does not do that. The boundary itself is genuine: "axis length governs which
construction to choose and nothing finer" is a falsifiable, pre-registered, replicated
statement about a method class (anatomical frames), and the matched-radius control on the
joint-level failure is a genuinely clean experimental design. It is a real contribution,
it is simply a small one, and the report's sin is only the title (Finding 4) and the
occasional "governs" (Finding 6).

---

## Attack on your planned defence of the baseline result

Your argument:

> The Kabsch-to-template baseline wins on the metric, but it cannot run the experiment
> this thesis is about. It has no anatomical axis, so there is no variable to hold fixed
> and vary, and the question of what governs frame consistency cannot be posed inside it.
> The anatomical frame is the instrument that makes the boundary measurable, not the
> result.

**Verdict: the argument is sound, and it is not special pleading — but the strongest
version of the objection is stronger than the version you have prepared for, and the
thesis survives it only if you state the objection yourself before the examiner does.**

Why it is sound: the objection "you built a worse method and redefined the contribution
to be the fact that you measured it" fails for one concrete reason — the *variable*.
The boundary experiment varies the length of an anatomical axis (hip axis vs shoulder
axis, joint set/scored set/constructor count held fixed). A template method has no such
axis; there is literally nothing to vary. The instrument claim is not rhetorical, it is
structural: you cannot ask "what governs frame consistency" inside a method that has no
frame to construct. This is the same relationship as "you cannot measure the orbit from
a telescope that has no pointing". That is a genuine limitation of the baseline *as an
instrument*, not a post-hoc redefinition.

The strongest version of the objection — the one to prepare against — is not "your
method loses" (conceded) but this:

> "Your boundary is a boundary *of a method that is dominated*. You have shown that
> within the family of anatomical frames, axis length is the design variable — but you
> have also shown that the entire family is dominated by a method outside the family.
> So the boundary you measured is a boundary of the *wrong family*. A practitioner does
> not care which axis to use inside a frame they should not be using at all. Your
> contribution is a detailed map of a country nobody needs to visit."

The thesis survives this only if the answer is: **the boundary is not about which frame
to deploy; it is about which geometric reasoning transfers to network-inferred joints.**
The claim is that (a) classical rigid-body reasoning about anatomical frames carries to
network joints at the level of construction choice, and (b) it stops at the finer levels
because articulation dominates. Both statements are about *the transfer of geometric
theory*, not about which normalizer to ship. The template's dominance is an engineering
fact about deployment; the boundary is a scientific fact about theory transfer. They are
different claims, and the report never conflates them. State that separation in the first
sentence of the answer and the objection loses its force.

What you must NOT do: (a) concede "the frame is not the best method, but…" and then
rest the contribution on the frame's value as a method — the objection wins that
exchange; (b) claim the template "is not a real baseline" — it is, and you pre-registered
it; (c) lean on "interpretable anatomical orientation" as the payoff — the report
explicitly says "nothing in this report measures whether that interpretability is worth
its cost" (§5.14.3), so the examiner will immediately ask you to justify it.

Recommended phrasing (fits in 30 seconds):

> "The baseline wins the deployment comparison, and I reported it before you had to ask.
> But the experiment this thesis is about cannot be run inside it. The boundary question
> requires an anatomical axis to hold fixed and vary — hip axis versus shoulder axis,
> everything else identical. Kabsch alignment has no axis; there is no variable. So the
> baseline answers which normalizer to ship, and the anatomical frame answers which
> geometric reasoning survives contact with a network-predicted skeleton. The report
> keeps those two claims separate, and the baseline result only ever touches the first."

---

## Final paragraph: mark band, and the single change that moves it up one

If I were the external examiner, I would place this in the **upper-second / first-class
borderline band — a strong 78-82 out of 100 in the departmental scale** — and I would
justify it exactly as the report frames itself: the method is not novel and is beaten by
its own baseline, but the pre-registration practice, the audit infrastructure, the
two-dataset/two-backbone replication of the central claim, and the systematic, documented
retraction of the author's own findings are well above the undergraduate norm, and the
boundary result, while small, is real, falsifiable and replicated. The single change that
would move it up one full band is not scientific but presentational: **fix the three
self-inflicted inconsistencies — the torso row missing from the circularity table
(Finding 1), the unreconciled 62.7 vs 75.3 mm (Finding 2), and the stale defence document
(Finding 3) — because all three are findable from the artifacts in minutes, and an
examiner who finds one will discount all the rigour that the rest of the report earns.**
If the defence goes well, that rigour is the thing that carries you over the line; the
science, honestly and accurately described, is enough for the band you want.
