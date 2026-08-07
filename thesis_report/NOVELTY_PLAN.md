# The novelty, honestly stated — and the two nights left

Defence: 9 Aug. This file is the plan. `SIR_CONCERNS.md` is the answer sheet.
`Minimal_Thesis_Report.tex` is the submission document. Read `RESEARCH_FINDINGS.md`
before opening your mouth at the viva — it is the adversarial sweep, and one
item in it (3DPCNet) changes your answers.

---

## What was done tonight (7 Aug) — two pre-registered experiments, both pass

The thesis's old weakness was that the answer to *"why keep the anatomical
frame when Kabsch wins?"* was rhetorical ("the baseline has no anatomical axis,
so it cannot pose the question"). Tonight gives it an **experimental** answer.

**Experiment 1 — `evaluation/anchor_corruption.py` (12th pre-registration).**
Corrupt the frame's support joints {hips, thorax}. Result, both backbones:

| σ (mm) | anatomical (XS) | template17 (XS) | anatomical (MB) | template17 (MB) |
|---|---|---|---|---|
| 0 | 53.45 | 43.30 | 44.13 | 40.96 |
| 20 | 71.95 | 43.70 | 64.57 | 41.38 |
| 40 | 103.91 | 44.87 | 99.40 | 42.61 |
| 80 | 215.66 | 49.15 | 217.81 | 47.15 |
| 160 | 337.87 | 63.22 | 339.86 | 61.63 |

The frame collapses 53 → 338 mm; the 17-joint fit moves 43 → 63 mm. The
four-joint control collapses with the frame, proving the failure is a property
of the **four-joint support**, not of the anatomical construction.

**Experiment 2 — `evaluation/selection_rule.py` (13th pre-registration).** A
fixed rule — *use the frame iff core confidence ≥ 0.7 AND distal confidence
< 0.7, else Kabsch* — across both corruption regimes:

- **Never worse than the better single alignment** at 19 of 20 (regime, σ,
  backbone) cells; trails by **5.4 mm in one transition cell** (tolerance 7).
- Beats Kabsch alone by **38.8 mm (XS) / 47.3 mm (MB)** at σ = 160 distal,
  bootstrap CIs excluding zero.
- Never routes into the collapsed arm under anchor corruption.
- Uses Kabsch at clean data (the better method there).

## The claim, one sentence (memorise)

> **The two training-free alignments have disjoint failure supports — distal
> corruption is invisible to the anatomical frame and visible to the template,
> anchor corruption is the reverse — and a confidence-gated routing rule that
> exploits this is never worse than the better single alignment at every
> severity tested, within a pre-registered 7 mm transition allowance, and beats
> Kabsch alone by 38–47 mm under severe distal corruption.**

## The anti-claims (say these before anyone else does)

1. **"The confidence signal is simulated."** True — the corruption experiments
   inject noise with no confidence channel; the rule's input models a detector
   whose per-keypoint confidence drops linearly with localization error. In
   deployment it consumes the detector's real per-joint confidence (YOLOv8
   keypoint scores). Say: *"the mapping from noise to confidence is a model, and
   I label it as one"*.
2. **"The frame still loses at clean data."** True, and the rule agrees: it
   uses Kabsch at clean.
3. **"The thresholds are tuned to the result."** False — fixed and
   pre-registered before running (commit `5dbc47a`, timestamps precede the
   results in `2a97b2e`).
4. **"One dataset, Gaussian noise, not real occlusion."** True — two backbones,
   one dataset, injected 3D noise. Real occlusion/detector failure is future
   work.
5. **"This doesn't make the frame win."** Correct — that is the point. The
   *combination* is never worse than the best single alignment.

## Why this is defensible as undergraduate novelty

- It is **not** a claim that the anatomical frame is better — the exact axis
  on which RESEARCH_FINDINGS said no novelty is available.
- It is a **map plus a rule**: the first systematic failure-support comparison
  of the two alignments (3DPCNet's geometric baseline is one table, no error
  bars, no corruption regimes), and a decision rule that falls out of it.
- It follows the thesis's own method: pre-registered before running, one
  backbone is not two, honest about the transition band.

## The two nights

**Night 1 (tonight, 7 Aug — mostly done).**
- [x] Anchor-corruption experiment, both backbones, pre-registered, Reading 1.
- [x] Routing experiment, both backbones, pre-registered, Reading 1.
- [x] RESULT.md for both; results committed after the pre-registrations.
- [ ] Add the two experiments as a short section of the minimal report (Ch. 5
      or a compact "post-freeze addendum" — draft included in
      `Minimal_Thesis_Report.tex`).
- [ ] Compile the minimal report, check page count, read it once aloud.

**Night 2 (8 Aug).**
- [ ] Rehearse the three answers from `SIR_CONCERNS.md` aloud, plus the
      3DPCNet answer and the Q4 answer ("I define cross-view agreement in my
      protocol; I have not established it is a named standard").
- [ ] Run `audit_numbers.py` (258/258) and `python -m unittest discover -s
      tests -q` (76+ tests) once more after the report edits.
- [ ] Send the supervisor email (`SUPERVISOR_EMAIL.md`) with the two-page
      `WORKFLOW.md` and the minimal report.
- [ ] Sleep. The viva is won on the honest answers, not on new numbers.

## Do NOT do (even with two nights)

- **Do not retrain anything.** CPU, no GPU results to show for it.
- **Do not claim a regime where the anatomical frame wins.** The occlusion
  pre-registration already burned that criterion; the routing rule is the
  honest version of the same idea.
- **Do not add more experiments.** The map is complete for this defence; every
  additional run is another number to defend.
- **Do not tune the routing threshold** to fix the 5.4 mm transition cell. That
  is tuning on the test split — the exact failure mode the pre-registrations
  exist to prevent.
- **Do not touch the frozen report's 258 claims.** The new experiments are
  recorded separately and cited by the minimal report only.
