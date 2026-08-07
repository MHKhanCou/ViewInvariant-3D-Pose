# Context prompt — paste this whole file into any model

Written 7 Aug 2026, two days before the defence. Everything below is verified
against stored artifacts by `evaluation/audit_numbers.py` (258/258 passing).
Where a claim is uncertain it says so.

---

## Who I am and what I am asking

I am an undergraduate CS student at Comilla University (ID 12108004). My thesis
defence is **9 August 2026**. My report is complete: 56 pages, compiled, 258
numerical claims automatically audited against stored result files, 76 unit
tests, seventeen pre-registered experiments committed to git before their experiments ran.

**I want to know whether any genuine research novelty is still reachable, and
whether my report should be cut further to make it more defensible.** I am
willing to change the pipeline if the case for it is strong. Tell me plainly if
you think that is a mistake.

---

## 1. My supervisor's brief, and exactly what he has seen

Sir gave me `Proposal.docx` on **18 Jan 2026**. Its direction: view-invariant 3D
human pose estimation. The problem it named — that the same person seen from
different cameras yields different 3D coordinates because each prediction lives
in its own camera's frame — is real and still open. His proposed mechanism was
**contrastive learning plus kinematic constraints**.

I submitted my own research proposal on **16 Feb 2026** committing to that
direction.

He then asked only to see output, and gave me
`https://github.com/KelvinHong/pose-estimation-3d`. **I showed him the output of
that repository. That is the entire extent of what he has seen.** He has not
seen MotionAGFormer, my method, or any result. He did not give me
MotionAGFormer — I found and chose it myself.

### The questions he has actually asked me

1. Where is the comparison between the old base model and my proposal?
2. How much did the output improve over base MotionAGFormer?
3. If it improved, what actually changed, and how?
4. How did I prove it, and against which dataset?
5. Why did my initial novelty fail? Why did the novelty in his proposal not work?
6. How did I approach this from the beginning? Which datasets, and why?

---

## 2. What I actually built

```
RGB video → YOLOv8 2D (frozen) → COCO→H36M → MotionAGFormer-XS (frozen)
          → body-frame canonicalization (0 params, 402 FLOPs) → cross-view comparison
```

The estimator is **completely frozen**. The one step I added rewrites the
predicted skeleton in a coordinate system built from the subject's own body:
`y = pelvis→thorax`, `x_raw = hip→hip`, orthonormalized by Gram-Schmidt. Because
both axes are read off the joints, they rotate with the body, so an unknown
camera rotation Q cancels exactly and never has to be estimated. No calibration.

**Second backbone:** MotionBERT, unmodified code, to check nothing was a
property of one network.

---

## 3. Results, and the one that goes against me

Human3.6M, 180 held-out camera pairs, cluster-bootstrapped over 30
subject-action groups.

| | Value |
|---|---|
| 3D accuracy (MPJPE), before and after | **45.149 mm → 45.149 mm, identical** |
| Cross-view joint distance, 13 non-constructor joints | **372.7 → 93.4 mm (−72.2 %)**, CI [+67.9, +75.5] |
| Pairs improved | 179 / 180 |
| Second backbone | −75.8 %, 180/180 |
| All 17 joints | 320.4 → 75.3 mm (−74.1 %); MotionBERT −77.5 % |
| Procrustes oracle floor (13 joints) | 56.2 mm; 87.0 % of the gap closed |

Accuracy **cannot** change — frozen estimator, zero added parameters. What
changes is *agreement between cameras*. The headline excludes the four joints
`{0,1,4,8}` the frame is built from, because the construction pins them (thorax
22.1 mm vs 197.5 mm for articulated joints), so a 17-joint average flatters me.

### The damaging result

A **Kabsch-alignment-to-one-fixed-template** baseline is training-free,
label-free, calibration-free and single-view — it meets *every* requirement I
claim as my framework's profile. It scores **57.5 mm against my 93.4 mm** and
wins on **180/180 pairs, both backbones, all 15 actions, three unrelated
templates, every centring tested.** It is in my abstract, the section "A Single-View Baseline, and It Wins", Limitations
and Conclusion. Its criterion was committed to git before it ran.

I have since pre-registered and run **two further searches for a regime where my
frame wins. Both failed:**

- **Distal-joint corruption.** Corrupt the eight joints past a hinge; my frame
  reads only four torso joints so it is exactly flat at 53.45 mm across every
  severity, while the baseline degrades 43.3 → 92.3 mm. But I required the
  crossover at ≤80 mm noise on *both* backbones; MotionBERT crossed at 40 mm,
  MotionAGFormer only at 160 mm. One backbone is not two. **Failed.**
- **Template body-proportion mismatch.** Retarget the template's limbs to 60 %
  (child-like against an adult template). The baseline moves from 57.47 to
  **57.59 mm — 0.12 mm, 0.2 %.** Both backbones. **Failed.**

**The mechanism, which is why I think a third attempt is futile:** Kabsch fits a
*rotation*, and both perturbations I tried are bilaterally *symmetric*. Shrinking
both arms, or corrupting both wrists, leaves the point cloud's principal axes
where they were. To move a Kabsch fit you need an **asymmetric** failure.

---

## 4. What collapsed, and why

I originally claimed four novelties. **All four collapsed, and I found every one
myself.**

1. **Training-free geometric canonicalization** → it is the **TRIAD algorithm**
   (Black 1964, spacecraft attitude determination). A literature-search failure,
   not a result failure. The primary-axis rule is Shuster & Oh 1981; the
   landmark-error propagation is Della Croce / Cappozzo biomechanics 1999, 2005.
2. **A geometric reliability score predicting pose error** → falsified five
   independent ways: ρ≈0 across simultaneous cameras; view selection straddling
   chance (4.78/8 vs 3.67/8); a cross-backbone **sign flip** (−0.707 MotionBERT,
   +0.375 plain MLP); reliability-weighted fusion worse than a plain mean
   (92.6 vs 88.5 mm); a pre-registered TTA test failing all three criteria.
3. **Bone-length consistency as an error predictor** → ρ = +0.492 on
   MPI-INF-3DHP, **+0.098** on Human3.6M. Retracted.
4. **Integrating all four** → the multi-scale variant builds each limb frame from
   exactly the three joints it is then scored on. Circular by construction.
   Demoted to exploratory.

**One shared cause:** geometric plausibility is invariant to a *coherent depth
error*. A skeleton wrong in depth but self-consistently so still has correct bone
lengths and plausible angles. Geometry cannot see the error because the error is
what geometry preserves. All four asked a geometric quantity to predict a
non-geometric failure.

### Why my supervisor's proposed mechanism did not become my contribution

It did not fail. Two things happened underneath it:

- **It was published while I worked.** **MoViD** (2026) does learned orthogonal
  motion-view disentanglement; **3DPCNet** (arXiv 2509.23455, Sept 2025) does
  estimator-agnostic post-hoc canonicalization of a frozen model — my exact
  problem statement. **V-VIPE** (CVPR-W 2024) uses a hip/spine body frame via
  Kabsch and calls it preprocessing. **Pr-VIPE** (2019) does calibration-free
  view-invariant embedding. Implementing his proposal in 2026 would reproduce
  published work. That the field went there in months says the direction was right.
- **It needs training.** Multi-view training at that scale needs compute and
  labels I do not have. What I could offer instead is the one thing those methods
  lack: no training, no labels, no camera parameters at any stage.

Note against me: 3DPCNet benchmarks a hand-built anatomical-landmark baseline of
exactly my family **and beats it**.

---

## 5. What actually survives as my contribution

A **three-level, pre-registered delimitation** of the classical principle that a
direction read from two joints distance L apart is uncertain by ~2σ/L, so longer
axis → more stable frame:

| Level | Question | Result |
|---|---|---|
| Between constructions | Does a longer axis give a better frame? | **Holds.** Shoulder axis beats hip axis on both backbones, intervals excluding zero |
| Within one construction | Which frame instance to trust? | **Fails.** The axis is an anatomical near-constant, p99/p1 = 1.29 |
| Between joints | Which joint will disagree? | **Fails, and backwards.** Torso-rigid joints sit at a *larger* radius yet disagree **2.5× less** than articulated ones |

Level 3 is the interesting one: a body is not a rigid body, and past a hinge the
estimator's error in the joint angle dominates the geometry entirely. Two of the
three tests are failures and **the boundary they establish is the contribution.**

Plus: seventeen pre-registered experiments, **seven failed their own criteria**, one returned
a competing method as better; 258 audited claims; 76 tests; a retraction of my
own error predictor.

---

## 6. Current state of the documents

Repository: `github.com/MHKhanCou/ViewInvariant-3D-Pose`

| File | What it is |
|---|---|
| `thesis_report/Minimal_Thesis_Report.tex` | The submission. 56 pages; main body 60, Ch5 22, then Appendix A/B and references |
| `thesis_report/WORKFLOW.md` | RGB→comparison walkthrough answering Sir's two questions, EN + BN |
| `thesis_report/SIR_QA.md` | All of Sir's questions plus likely follow-ups, EN + BN |
| `thesis_report/SUPERVISOR_EMAIL.md` | The email to send before he reads the report |
| `thesis_report/DEFENSE_QA.md` | Long viva prep: timed pitches, hard questions |
| `thesis_report/EXPLAIN.md`, `EXPLAIN_SIMPLE.md` | Plain-language, EN + BN |
| `REPO_MAP.md` | Every module, its artifact, its report section |
| `thesis_artifacts/*/PREREGISTRATION.md` | Eleven, each committed before its run |
| `thesis_artifacts/{occlusion,mismatch}/RESULT.md` | The two post-freeze failures |

Chapter 5 was 56 pages and held every contribution, so ten supporting sections
were moved verbatim into a new Appendix B rather than deleted — main body 73 → 60,
Ch5 35 → 22, all 304 claims intact, no broken cross-references.

Title, **not confirmed by Sir** — I must ask him: *"A Lightweight, Training-Free,
Reliability-Aware Geometric Canonicalization Framework for Cross-View
Comparability of Frozen Monocular 3D Pose Predictions."* I doubt
"Reliability-Aware", since the report falsifies that score as an accuracy
predictor five ways and it survives only against a narrower target.

---

## 7. What I want from you

Answer these directly, and disagree with me where you think I am wrong.

1. **Is there reachable novelty in two days that does not require retraining?**
   I believe the only untested axis is **asymmetric** failure — unilateral
   occlusion, single-limb dropout, hemiplegic or amputee gait — because that is
   the one thing that actually rotates a Kabsch fit. Is that worth running, or am
   I chasing a result I have already failed to find twice?
2. **Should I change the pipeline?** Be blunt. I have a compiled report with 258
   audited claims and a defence in two days.
3. **How much further should the report be cut, and what specifically?** The goal
   is defensibility — fewer pages I must be able to answer for.
4. **Is a pre-registered negative-results / boundary-mapping thesis a strong
   undergraduate contribution, or does it read as a failed project?** How should I
   present it so it reads as the former?
5. **Is my answer to "a simpler method beats you" good enough?** It is currently:
   *Kabsch cannot run the experiment this thesis is about — it has no anatomical
   axis, so there is no axis to hold fixed and vary, and the question of what
   governs frame consistency cannot be posed inside it.* Is that a real defence or
   a rhetorical dodge?

### Constraints — do not suggest violating these

- No retraining. No camera calibration. The estimator stays frozen.
- Never claim improved *accuracy*. The claim is cross-view *comparability*.
- Never call the prediction-vs-prediction metric MPJPE; it is *cross-view joint
  distance*.
- Cannot claim "model-agnostic" — only two backbones tested.
- Do not propose reviving the 67 %, 40 %, or 28.4 % figures. All are invalid.
- Report failures as findings. Do not help me spin one.
