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

**Experiment 3 — `evaluation/misdetect_invariance.py` (14th pre-registration).**
The end-to-end question an examiner will ask: does the corruption matter
through the *real* detection path, and is there a real confidence signal to
gate on? Re-detected MPI-INF-3DHP S1/Seq1 cams 0–1 with real YOLOv8, displaced
the 2D keypoints of distal and core joints by up to 434 px (0.6 in normalized
input space), re-lifted with the frozen lifter:

| condition | cam0 | cam1 |
|---|---|---|
| distal f=0.15 (434 px) | 0.21 mm | 0.25 mm |
| core f=0.15 | 0.32 mm | 0.33 mm |
| detector confidence (mean) | 0.74 | 0.81 |
| frames with confidence < 0.9 | 100% | 100% |

**Reading 1.** The lifter is invariant to 2D keypoint error at any magnitude
(≤ 0.33 mm vs 53 → 338 mm for the same joints corrupted at the 3D level), and
the real confidence channel is flat (all frames < 0.9 — no threshold can
separate frames on clean data). The failure surface is at the 3D alignment
level, end to end, and the measured signal that actually varies is the
analytic reliability score. (One correction was documented in the
pre-registration: the cached `components[:,0]` is a reliability component, not
detector confidence.)

## The claim, one sentence (memorise)

> **The two training-free alignments have disjoint failure supports — distal
> corruption is invisible to the anatomical frame and visible to the template,
> anchor corruption is the reverse — and a confidence-gated routing rule that
> exploits this attains the better single alignment's aggregate-mean
> cross-view distance at 19 of 20 (regime, severity, backbone) cells, trails
> it by 5.4 mm in the single transition cell (within a pre-registered 7 mm
> allowance), and beats Kabsch alone by 38–47 mm under severe distal
> corruption — evaluated under a simulated, noiseless confidence signal that
> is labelled as such.**

## The anti-claims (say these before anyone else does)

1. **"The confidence signal is simulated."** True — the corruption experiments
   inject noise with no confidence channel; the rule's input models a detector
   whose per-keypoint confidence drops linearly with localization error. In
   deployment it consumes the detector's real per-joint confidence (YOLOv8
   keypoint scores). Say: *"the mapping from noise to confidence is a model, and
   I label it as one. The fourteenth experiment measured the real channel: on
   clean data it is flat (every frame below 0.9), so the gate is a stand-in for
   a channel that carries no usable signal — the signal that does vary is the
   analytic reliability score"*.
2. **"The frame still loses at clean data."** True, and the rule agrees: it
   uses Kabsch at clean.
3. **"The thresholds are tuned to the result."** False — fixed and
   pre-registered before running (commit `5dbc47a`, timestamps precede the
   results in `2a97b2e`).
4. **"One dataset, Gaussian noise, not real occlusion."** True — two backbones,
   one dataset, injected 3D noise. Real occlusion/detector failure is future
   work.
5. **"This doesn't make the frame win."** Correct — that is the point. The
   *combination* is never worse, in aggregate mean, than the best single
   alignment.
6. **"You picked the threshold 0.7 to fit the crossovers."** The threshold was
   fixed and committed before this experiment ran (`5dbc47a`), and it is
   deliberately conservative — it loses 5.4 mm on XS in the transition band
   rather than being tuned to win. Tuning it per backbone to the known
   crossovers (40 vs 160 mm) would be the exact tuning-on-test failure the
   pre-registrations exist to prevent.
7. **"The routing isn't per-frame."** Correct — in this controlled experiment
   the decision is per corruption level (confidence is constant within a
   level). Per-frame deployment with noisy real detector confidence is the
   assumption the rule makes, not something the synthetic experiment measures.
8. **"Real detection errors would break your map."** Measured, not assumed
   (Experiment 14, Reading 1): 2D keypoint displacement up to 21% of the frame
   width moves the lifted pose ≤ 0.33 mm — the 2D channel cannot create a 3D
   corruption regime, so the map is confined to the 3D level. What is *not*
   covered: missed person, truncation, motion blur (stated as out of scope).

## Why this is defensible as undergraduate novelty

- It is **not** a claim that the anatomical frame is better — the exact axis
  on which RESEARCH_FINDINGS said no novelty is available.
- It is a **map plus a rule**: the first systematic failure-support comparison
  of the two alignments (3DPCNet's geometric baseline is one table, no error
  bars, no corruption regimes), and a decision rule that falls out of it.
- It follows the thesis's own method: pre-registered before running, one
  backbone is not two, honest about the transition band.

## The two nights

**Night 1 (tonight, 7 Aug — done).**
- [x] Anchor-corruption experiment, both backbones, pre-registered, Reading 1.
- [x] Routing experiment, both backbones, pre-registered, Reading 1.
- [x] 2D-input invariance experiment (14th), real detection path, Reading 1;
      P2 amended with a documented measurement correction.
- [x] RESULT.md for all three; results committed after the pre-registrations.
- [x] All three in the minimal report's post-freeze addendum (§5.8);
      report recompiled clean.
- [ ] Read the compiled report once aloud.

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
- **Do not add any more experiments.** The map is complete for this defence
  (seventeen pre-registered experiments, three post-freeze and established); every
  additional run is another number to defend. Experiment 14 closed the 2D
  channel — there is nothing left to measure tonight.
- **Do not tune the routing threshold** to fix the 5.4 mm transition cell. That
  is tuning on the test split — the exact failure mode the pre-registrations
  exist to prevent.
- **Do not touch the frozen report's 304 claims.** The new experiments are
  recorded separately and cited by the minimal report only.
