# Sir's concerns — what they are, and the exact answers

Every number here is recomputed from stored artifacts by
`evaluation.audit_numbers.py` (304/304 pass) or is a post-freeze pre-registered
experiment recorded in `thesis_artifacts/`. English first, Bengali at the end.

---

## Concern 1 — "Where is the comparison between old base and your proposal?"

This question has two readings, and both need an answer.

### Reading A: base estimator output vs the proposed canonicalized output

This comparison **is in the report** — it is the whole of Chapter 5 — but it is
easy to miss because the answer has three rows that must be read together, and
the raw numbers live in three artifacts:

| What is compared | Before (base) | After (proposal) | Where it lives |
|---|---|---|---|
| 3D pose accuracy (MPJPE) | 45.149 mm | **45.149 mm — unchanged** | `baseline_results.json`, §5.1 |
| Cross-view distance, 13 non-constructor joints | 372.7 mm | **93.4 mm (−72.2 %, CI [67.9, 75.5])** | `noncon/noncon.json`, §5.3 |
| Cross-view distance, all 17 joints | 320.4 mm | 75.3 mm (−74.1 %) | `h36m_crossview/`, §5.3 |
| Per-frame Procrustes oracle (floor) | — | 56.2 mm; 87.0 % of the gap closed | `noncon/`, §5.3 |
| Second backbone (MotionBERT) | 320.4 mm | 75.8 % improvement, 180/180 pairs | `h36m_motionbert/`, §5.8 |
| **A simpler baseline beats the proposal** | Kabsch-to-template: **57.5 mm** vs our 93.4 mm | wins 180/180 pairs, all 15 actions | `template/template.json`, §5.6.1 |

The figure that shows it visually: `thesis_artifacts/figures/fig_realview.png`
(raw side by side vs canonical side by side, real cameras).

**The one sentence to say:** *"Accuracy is identical by construction — the
estimator is frozen and canonicalization adds zero parameters. What changes is
agreement between cameras: 372.7 mm → 93.4 mm. And I ran the comparison a
reader would ask for, Kabsch alignment to a fixed skeleton, and it beats mine,
57.5 mm; that is in the abstract, not hidden."*

### Reading B: the original proposal's pipeline vs the delivered pipeline

The proposal (Feb 2026) promised a **trainable** pipeline: 2D detector →
canonicalization (re-centre + torso alignment in 2D) → **MLP lifting model**
→ bone-length loss. The delivered system is different: 2D detector →
**frozen MotionAGFormer/MotionBERT** → **3D body-frame canonicalization**
(post-hoc, zero parameters). The change happened for a documented reason:
training a model of our own would have been weaker than a published backbone,
so no gain would be measurable, and freezing is what makes the result
transferable. `PROPOSAL_TO_DELIVERY.md` maps every proposal objective to its
delivered outcome. The early MotionBERT-vs-MLP experiments that predate the
final pipeline (MLP 70.6 mm vs pretrained MotionBERT 180.9 mm on
MPI-INF-3DHP) are archived at the repository root — they belong to the
superseded domain-adaptation direction and are **not** part of the final
thesis.

**Say:** *"The proposal's pipeline changed. I deliver the same two components —
canonicalization and geometric analysis — but applied to a frozen published
backbone instead of a trained MLP, because a trained model of my own could not
have been measured against the state of the art. The comparison you asked about
is in Chapter 5: base output against canonicalized output, and against the
Kabsch baseline."*

---

## Concern 2 — "too long for an undergraduate report, where is the novelty?"

Two separate things are being objected to, and they need two answers.

**The length.** This is resolved. The submission is the **Minimal Thesis Report**
(`Minimal_Thesis_Report.tex` → `Thesis_12108004.pdf`, **55 pages**): full report
structure, every headline number, every honest negative result, none of the
scaffolding. The earlier long version survives as the **97-page technical
supplement** (`Thesis_12108004_supplement.pdf`), kept for the record and attached
only if he wants the complete evidence.

**The novelty.** Say the honest thing, which is now stronger than it was two
days ago:

1. **The method is not claimed as novel.** The frame is TRIAD (Black 1964); the
   error propagation is biomechanics (Della Croce et al. 1999, 2005); the
   closest prior work, 3DPCNet (ICASSP 2026), publishes this method *family* as
   a baseline. All of this is stated in the report and the README.
2. **The contribution is the experimental boundary**, established by
   pre-registered tests: axis length governs the choice *between* frame
   constructions and nothing finer (the multi-scale figure 55.1 % was demoted
   when found circular; the bone-length signal retracted on the second dataset;
   the reliability score falsified five ways as an accuracy predictor and shown
   to gate canonicalization quality instead).
3. **NEW — two post-freeze pre-registered experiments (12th and 13th) give the
   thesis an experimental answer to "why keep the frame at all".** They map the
   failure supports of the two alignments:
   - **Distal corruption** (occlusion protocol): the frame is *exactly flat*
     (53.45 mm at every severity — it never reads the corrupted joints), while
     Kabsch degrades 43.3 → 92.3 mm.
   - **Anchor corruption** (new): corrupt the frame's own joints
     {hips, thorax} and it collapses 53.45 → 337.9 mm, while the 17-joint
     Kabsch fit degrades gracefully 43.3 → 63.2 mm.
   - The failure modes are **disjoint**. A fixed confidence-gated routing rule
     (use the frame when the core is reliable and the periphery is not;
     otherwise Kabsch) is **never worse than the better single alignment** at
     19 of 20 (regime, severity, backbone) cells, trails by 5.4 mm in one
     transition cell, and beats Kabsch alone by 38–47 mm under severe distal
     corruption. Pre-registered before running; both Reading 1.
   See `thesis_artifacts/anchor_corruption/` and `thesis_artifacts/selection/`,
   and `NOVELTY_PLAN.md` in this directory.

**If he asks "is that enough novelty for an undergraduate thesis?"** — yes, and
it is rarer than a manufactured method: seventeen pre-registered experiments in version
history, seven failed, an automated audit of 304 claims, 76 tests. The
contribution is the boundary map plus the routing rule, not a claim that the
frame wins.

---

## Concern 3 — "Someone has already published your frame as a losing baseline"

Raise it before he does (see `RESEARCH_FINDINGS.md`, section 1 — 3DPCNet,
ICASSP 2026, §3.3). The answer has three qualifications and one admission:

> "3DPCNet publishes a two-vector anatomical baseline of my family and reports
> it losing to their learned module by ~6× on rotation error. Three things
> qualify it: it is their own self-implemented baseline with no error bars; it
> measures alignment to a ground-truth canonical pose while I measure agreement
> between two cameras; and its primary axis is the shoulder–hip plane normal,
> which is not my construction — TRIAD-family, not identical. But the direction
> agrees with my finding, and I would rather say the evidence points the same
> way twice."

---

## Concern 4 — the word "Reliability-Aware" in the title (RESOLVED — removed)

**This question can no longer be asked. The word is gone.** The title is now:
*A Lightweight, Training-Free Geometric Canonicalization Framework for
Cross-View Comparability of Frozen Monocular 3D Pose Predictions.*

Do not rehearse a defense of the old word. Volunteer the removal instead, which
is the stronger move:

> *"I removed 'Reliability-Aware' from the title myself. The report falsifies
> that score as an accuracy predictor five different ways, so leaving the word
> in the most prominent line of the document would have overstated a component
> the body largely disproves. What survives is narrower and is reported as
> exactly that: the score gates canonicalization quality — the function its own
> specification names — on both backbones, with the confound controlled. That
> finding is in Chapter 5; it is not a title claim."*

The point to land: the title was corrected because the evidence did not support
it, not because someone objected.

---

## Concern 5 — "Are you sure these numbers are real?"

> "Open the git log, Sir. Seventeen pre-registered experiments, each criterion committed
> before its experiment ran, seven failed. Nobody fabricates failures. Then run
> `audit_numbers.py` — it recomputes all 304 claims from stored files and fails
> if one drifts."

---

## What changed tonight (7 Aug) — the two new experiments

| | `occlusion/` (10th, failed) | `anchor_corruption/` (12th) | `selection/` (13th) |
|---|---|---|---|
| Corrupted joints | distal {2,3,5,6,12,13,15,16} | anchors {1,4,8} | — |
| Anatomical frame | flat 53.45 mm at all σ | collapses 53.45 → 337.9 mm | routed — never worse |
| Template Kabsch | degrades 43.3 → 92.3 mm | graceful 43.3 → 63.2 mm | routed — never worse |
| Reading | 3 (failed, σ ≤ 80 both backbones) | **1 (established)** | **1 (established)** |

Both new pre-registrations were committed before running (`5dbc47a`); results
after (`2a97b2e`).

---

# বাংলা (সংক্ষিপ্ত)

**স্যারের প্রশ্ন ১ — "পুরোনো base আর আমাদের proposal-এর তুলনা কোথায়?"**
উত্তরটা Chapter 5-এ আছে, তিন লাইনে: নির্ভুলতা (MPJPE) আগে-পরে ৪৫.১৪৯ mm — অপরিবর্তিত;
cross-view দূরত্ব ৩৭২.৭ → ৯৩.৪ mm (−৭২.২%); আর একটা সহজ Kabsch baseline আমাদের
হারায় (৫৭.৫ বনাম ৯৩.৪ mm) — এটা abstract-এ লেখা। চিত্রটা `fig_realview.png`।

**স্যারের প্রশ্ন ২ — "৯০ পৃষ্ঠা, novelty কোথায়?"** দুই উত্তর: (ক) রিপোর্ট
সংক্ষিপ্ত হচ্ছে — নতুন **Minimal Thesis Report** (~২০ পৃষ্ঠা) জমা দেবেন;
(খ) novelty-র সৎ অবস্থান: method টা TRIAD (১৯৬৪) থেকে নেওয়া, তাই দাবি করা হয় না;
contribution হলো pre-registered পরীক্ষাগুলোর সীমা-মানচিত্র। **আজ রাতে দুইটা নতুন
pre-registered পরীক্ষা হয়েছে**: anchor joint নষ্ট করলে frame ভেঙে পড়ে
(৫৩→৩৩৮ mm), আর একটা fixed নিয়ম (core ঠিক থাকলে আর periphery ভাঙা থাকলে frame,
নইলে Kabsch) কখনোই সেরা single alignment-এর চেয়ে খারাপ নয় এবং severe distal
corruption-এ Kabsch-এর চেয়ে ৩৮–৪৭ mm ভালো। দুটোই Reading 1, দুই backbone-এ।
