# Final roadmap — 4 Aug 2026 onward

Report due 6 Aug. Defense 9 Aug. Deliverable A is done and committed (`63ddd6e`).

## MUST DO

| # | Item | Time | Thesis impact | Defense impact | Sci. risk | Eng. risk |
|---|------|------|---------------|----------------|-----------|-----------|
| 1 | Rehearse the abstention finding aloud; add it to DEFENSE_QA as Q7 | 40 m | none (done) | **high** | none | none |
| 2 | Two-view canonical viewer in `app.py` | 2.5 h | low | **high** | none | medium |
| 3 | Send report + email supervisor re: title wording | 30 m | — | high | none | none |

**Item 1 is the highest-value 40 minutes available.** The examiner's sharpest
question is "your title says Reliability-Aware and Chapter 5 destroys the
reliability score." Until today the only answer was a retraction. Now the
answer is a result: the score fails at predicting pose error and succeeds at
gating canonicalization, on both backbones, with the confound controlled. The
distinction must be delivered in one sentence without hedging.

**Item 2 is the only demonstrator worth building** — see below.

## NICE TO HAVE

| # | Item | Time | Thesis impact | Defense impact | Sci. risk | Eng. risk |
|---|------|------|---------------|----------------|-----------|-----------|
| 4 | Coverage–error figure for the triage result | 45 m | medium | medium | none | low |
| 5 | Cite CHAMP / CUPS, one paragraph on why not conformal | 30 m | medium | medium | none | none |
| 6 | View-count curve — data already in `fusion_results.json` | 20 m | low | low | none | none |

Item 5 preempts a specific question: *"conformal prediction gives you a coverage
guarantee — why didn't you use it?"* The answer is real and short (conformal
needs a labelled calibration set and targets pose error; this is label-free and
targets frame quality), and the report currently gestures at it without citing
anyone.

## DEMO ONLY — do not put in the report

| # | Item | Time |
|---|------|------|
| 7 | Frame-construction slider (global / torso / limb / shoulder) | 2 h |
| 8 | BVH → Blender retarget | — |

Item 8 is not possible: Blender is not installed and the BVH path is already
tested. Item 7 is a good demo and a bad use of the last day.

## DO NOT DO

- **Cross-view retrieval.** It was item 2 of the original plan and it should be
  cut. It needs a gallery protocol, a per-backbone run, and a careful
  non-comparison to Pr-VIPE, and a null result on the last day buys a fourth
  negative in a report that already carries three. The §3.7/§5.14 contradiction
  it was going to fix should instead be fixed by one sentence marking §5.14 as
  superseded protocol — 10 minutes, same outcome.
- **Any pipeline change.** 131 audit claims are frozen against artifacts.
- **Rewriting prose to lower an AI-detector score.** Would misrepresent
  authorship. Check the university policy and disclose if required.
- **A second conditioning variant.** The criterion failed; trying variants until
  one passes is the exact failure mode the pre-registrations exist to prevent.

---

# Deliverable B — the demonstrator

`app.py` is Gradio, single-input, image and video. Four options were considered.

| Idea | Effort | Impact | Research value |
|------|--------|--------|----------------|
| **Two-view canonical viewer** — same instant from two cameras, raw side by side, canonical side by side, live cross-view distance and the oracle floor | 2.5 h | high | **Demonstrates the central claim directly.** The thesis's headline number is a distance between two canonicalized predictions; nothing currently *shows* that. |
| **Abstention overlay** — run the reliability gate per frame, grey out abstained frames, show the running mean with and without | 1.5 h | high | Demonstrates today's finding, which is the newest thing in the thesis. |
| Frame-construction slider | 2 h | medium | Demonstrates the axis-length principle, which is already the clearest figure in the report. |
| Oracle-gap bars | 30 m | low | The figure already exists. |

**Recommendation: the two-view viewer, with the abstention overlay folded into
it as a checkbox.** One build, both results. The viewer answers "does
canonicalization do anything" and the checkbox answers "when should you not
trust it" — which is exactly the arc the report now takes. Build the viewer
first; if it runs long, ship it without the checkbox rather than shipping half
of each.

Engineering risk is medium, not low: the existing app has no notion of two
synchronized inputs, so this is new plumbing, not a new tab. Budget the 2.5 h
honestly and abandon at the halfway mark if the plumbing fights back — the
report does not depend on it.

---

# Deliverable C — final literature search

**Question asked: is there a small overlooked contribution implementable in
under a day?**

**Answer: no.** Reported plainly because the honest answer is more useful than a
manufactured one.

What the search found:

- **Canonicalization is occupied.** 3DPCNet (arXiv 2509.23455, ICASSP 2026) does
  learned estimator-agnostic canonicalization; already cited, tabulated and
  contrasted in §2. The canonical-domain taxonomy paper (arXiv 2501.16146)
  covers body-fixed vs partially-body-fixed vs global frames. Nothing here is
  claimable and nothing is missing from the report.
- **Abstention and selective prediction are active but aimed elsewhere.** The
  2025–2026 work is overwhelmingly on language models. In pose, the uncertainty
  line is conformal: CHAMP (arXiv 2407.06141) conformalizes multi-hypothesis 3D
  pose, CUPS (arXiv 2412.10431) conformalizes pose-shape with a deep uncertainty
  function and handles non-exchangeability. **Both require a labelled
  calibration set and both target pose error.** Neither targets whether a
  canonical frame is fit to use, which is the target that worked today. That gap
  is real, and it is a masters-scale project, not a one-day one — it needs a
  calibration protocol, a coverage guarantee, and a downstream task.
- **Geometric conditioning as an uncertainty proxy** remains established in
  adjacent fields (multiview condition numbers, LiDAR localizability) and, per
  today's negative result, does not transfer per-frame to body-frame
  canonicalization. That negative is now the report's own finding.

The only sub-day action the search justifies is a citation, not a contribution:
item 5 above.

---

# Deliverable D — critical review, no softening

**The four real weaknesses.**

1. **The core mechanism is a coordinate transform.** Gram-Schmidt body frames
   predate this thesis by decades. The defense — frozen predictions, zero
   parameters, an exactness proof, and a requirement profile no learned method
   matches — is genuine but it is a *positioning* defense, not a technical one. A
   reviewer who does not accept the positioning will not find a technique here.
   This is correctly characterized in the report and should not be oversold in
   the viva.

2. **The headline metric measures agreement, not correctness.** Cross-view
   distance between two canonicalized predictions falls if both predictions are
   wrong in the same way. The Procrustes oracle floor and the retained
   correlation with ground-truth error (+0.610 vs +0.601, §5.7) address this, but
   only partially, and the correlation check exists on MPI-INF-3DHP alone.
   **This is the attack I would lead with as an examiner.** Prepare it.

3. **No downstream task succeeds.** The retrieval experiment is negative and its
   protocol is superseded. So the thesis improves a metric and never shows the
   metric buys anything. Every other weakness is survivable; this one is the
   honest ceiling on the work's significance, and it is why the contribution is
   a solid undergraduate thesis and not a workshop paper without further work.

4. **Two datasets, both lab multi-camera rigs; two backbones.** "Model-independent"
   rests on n=2. "Dataset-independent" rests on two captures that share a
   modality, a joint convention, and a controlled setup. No in-the-wild
   validation exists.

**What is genuinely strong, stated without inflation.** The negative results are
real science and there are now four of them, three pre-registered before the
number existed, with the pre-registrations in version history. The bone-length
retraction removed the project's most attractive claim on cross-dataset evidence
rather than defending it. The multi-scale left/right error was self-found and
disclosed. 131 numbers trace to artifacts and 67 tests pass. That discipline is
the thesis's actual contribution to a reader, and it is rarer at this level than
any result would be.

**On today's finding specifically.** It is the most interesting thing in the
report and it is also the least protected. It was a comparator in the
pre-registration, not its subject, so it is exploratory. It replicates on two
backbones and survives one confound control, which is more than the bone-length
signal ever had — and the bone-length signal still died. State it as exploratory
every time it is stated, and do not let it drift into the abstract as a headline.
