# Paper Plan

Derived from a 104-agent literature sweep (2026-08-03, adversarially verified;
raw findings in the workflow transcript) reconciled with our own results,
including the **view-selection claim we withdrew the same day**.

Read this with `FINAL_REPORT.md` (evidence) and `planning/DEFENSE_QA.md`.

---

## 1. What the literature sweep changes

### Occupied — do NOT claim these

| Claim we might have made | Who already owns it | Detail |
|---|---|---|
| "Training-free post-hoc canonicalization of a frozen monocular 3D pose estimator" | **3DPCNet** (arXiv 2509.23455, Sept 2025) | Same problem statement, estimator-agnostic, operates directly on 3D joints. **And it benchmarks a hand-built anatomical-landmark baseline of our family — and beats it: 62.9–64.6 mm / 20.6–21.6° vs 3DPCNet 47.6–47.7 mm / 3.4–4.2° on MM-Fi.** |
| "Body frame from hips/shoulders/spine" | **V-VIPE** (CVPR-W 2024) | Aligns left hip, right hip, spine via Kabsch — analytic, zero learned parameters — and treats it as *preprocessing*, not a contribution. |
| "Calibration-free cross-view pose comparability" | **Pr-VIPE** (2019/2020) | Staked this framing years ago. |
| "Gram-Schmidt for view-invariance" | **MoViD** (SenSys 2026) | Textbook Gram-Schmidt orthogonalization — on motion *features* against a learned view basis. Also appears in CMANet via 6D rotation. Our phrasing must always be qualified: *training-free, applied to frozen 3D joint outputs*. |
| "Post-hoc uncertainty for a frozen pose model" | **Conformal keypoint detection** (CVPR 2023 Highlight), **CHAMP**, **CUPS** | Conformal prediction already wraps frozen pose/keypoint models with coverage guarantees. |
| The supervisor's original proposal (contrastive cross-view + kinematic constraints) | **MoViD**, **3DPCNet** | MoViD pairs contrastive cross-view alignment with SMPL canonical anchors; 3DPCNet trains self-supervised on synthetically rotated views — the exact proposed augmentation. Not revivable as a standalone contribution; use as a **contrast arm / related work**. |

### The seam that is real

Every canonicalization competitor **requires training**; every uncertainty
competitor **requires a labeled calibration set**:

- MoViD: end-to-end trained on H36M + 3DHP + InstaVariety + AMASS with GT SMPL supervision
- V-VIPE: VAE trained with 3D GT supervision
- CMANet: self-supervised training **and** multi-view input at inference
- 3DPCNet: self-supervised training on synthetically rotated poses
- Conformal methods: inductive CP definitionally needs a held-out **labeled** calibration set and a chosen miscoverage level

Ours needs **no training, no labels, no camera parameters, no multi-view input
at inference**. That requirement profile is the paper's strongest asset.

### Verifier warnings we must obey

The adversarial verifiers **killed** every "therefore X is unclaimed" leap
(0-3 and 1-2 votes). Absence of a mechanism in four specific papers is *not*
global novelty. Before any "first to" sentence we must systematically search:
**learned view selection, active view selection, epipolar-free / triangulation-
free multi-view fusion, part-based limb-frame normalization.** Until then the
paper claims a *requirement profile*, never priority.

Two citation defects to fix before submission:
1. The quote *"learning a small neural network to perform canonicalization is
   better than using predefined heuristics"* belongs to **Kaba et al.
   arXiv:2211.06489 (ICML 2023)**, not 2405.14089 — and its argument (PCA /
   inertia-tensor sign and ordering ambiguity) **does not transfer** to a
   landmark-defined frame where semantics fix the sign. Cite **3DPCNet Table 1**
   instead as the in-domain learned-beats-geometric prior.
2. The circulating V-VIPE hip/spine and MoViD SMPL sentences are paraphrases,
   not verbatim. Quote only the verified HTML/ar5iv text.

---

## 2. Headline contribution (reconciled with our own retraction)

The sweep proposed "reliability driving multi-view **selection**/fusion". We
**disproved selection** on 2026-08-03 (within-frame ρ ≈ 0; picked-view rank
4.78/8 on S1 vs 3.67/8 on S2 — straddling chance). The plan therefore drops
selection and headlines what survived:

> **A label-free, analytic reliability signal for frozen monocular 3D pose
> predictors — and what it can and cannot do.** It enables calibration-free
> multi-view fusion in a shared analytic body frame (+8–24% over an arbitrary
> view across four subject/motion conditions) and reliable corruption-triggered
> abstention (ρ = −0.813) — but it does **not** rank simultaneous clean views by
> accuracy (within-frame ρ ≈ 0). Geometric plausibility detects corruption, not
> viewpoint-induced depth error.

Canonicalization is the **enabling substrate**, never the claim. The negative
result is a first-class contribution, not an appendix: it is the part no
competitor reports, and it is what makes the positive claims credible.

Component ranking for the paper:

| Component | Status | Role in paper |
|---|---|---|
| 2 · Label-free analytic reliability + hard gates | **Strongest seam** (all competitors need labeled calibration) | Core |
| 5 · Calibration-free multi-view **fusion** | Replicates 4/4 conditions | Core |
| — · The delimitation (corruption ≠ depth error) | Ours alone; nobody reports where their signal fails | Core |
| 4 · Multi-scale per-limb frames | Least-contested built component (every cited baseline is a *single global* frame) | Secondary |
| 1 · Gram-Schmidt canonicalization | **Occupied and empirically beaten by 3DPCNet** | Substrate only — never a claim |
| 3 · Bone-length as analytic metric | Adjacent to BLAPose/DDHPose (training losses) | Supporting metric |

---

## 3. Table 1 — the requirement matrix (the paper's spine)

| Method | Training | Labels | Camera params | Multi-view at inference | Frozen backbone | Uncertainty |
|---|---|---|---|---|---|---|
| MoViD (SenSys '26) | ✓ | GT SMPL | — | — | ✗ | ✗ |
| 3DPCNet ('25) | ✓ (self-sup) | synth rotations | — | — | ✓ | ✗ |
| V-VIPE (CVPR-W '24) | ✓ (VAE) | 3D GT | — | — | ✗ | ✗ |
| CMANet ('24) | ✓ | — | — | **required** | ✗ | keypoint conf. only |
| Conformal KP (CVPR '23) / CHAMP / CUPS | ✗ | **labeled calib. set** | — | — | ✓ | ✓ (coverage) |
| **Ours** | **✗** | **✗** | **✗** | optional | **✓** | **✓ (analytic, no guarantee)** |

Last row's honesty is the point: we offer **no coverage guarantee**. State that
explicitly (see §5).

---

## 4. Baselines and ablations reviewers will demand

Ranked; ✅ = already have.

1. **3DPCNet as learned comparator** — public code, the natural head-to-head. Expect to lose on accuracy; the argument is the requirement profile, not the number. *Not optional — a reviewer will ask.*
2. **Single global frame vs multi-scale** ✅ (isolates the per-limb gain: +36% on 29/29 pairs)
3. **Conformal-calibrated abstention threshold** alongside the analytic gate — the protocol reviewers already know for frozen-model uncertainty.
4. **Best-fixed-camera and GT-oracle-view bounds** ✅ (90.2 mm / 87.9 mm)
5. **Naive averaging** ✅ and **calibrated triangulation** (we have full MPI calibration — this is the "what you give up" reference)
6. **Second frozen backbone (MotionBERT)** — highest-value zero-training experiment; checkpoint already local
7. **Risk-coverage / AURC + AUROC** reporting for abstention, alongside our Spearman numbers

---

## 5. Experiment plan (no retraining, local data only)

| # | Experiment | Why | Cost |
|---|---|---|---|
| 1 | **MotionBERT as second frozen backbone** | Converts "model-independent by construction" into evidence; the one claim our examiner explicitly barred. Checkpoint is local (162 MB). | ~1 day (needs a MotionBERT lift path) |
| 2 | **Calibrated triangulation reference** | Quantifies the cost of being calibration-free. Full MPI calibration already parsed in `canonical/mpi_eval.py`. | ~half day, numpy on cache |
| 3 | **Conformal abstention calibration** | Meets reviewers on their protocol; contrast label-free analytic gate vs conformal set with a labeled split, and report the coverage we forfeit. | ~half day |
| 4 | **3DPCNet head-to-head** | Public code; the required comparator. | 1–2 days incl. setup |
| 5 | **Risk-coverage/AURC + AUROC** for abstention | Standard selective-prediction reporting. | hours |
| 6 | **Systematic novelty search** (view selection, triangulation-free fusion, part-based limb frames) | Required before any priority sentence. | ~half day |

**Deliberately excluded:** any retraining; reviving the contrastive branch
(published); more MPI sequences (breadth without new claims).

---

## 6. Venue

⚠️ **Unevidenced — judgment only.** The sweep found no verified sources on
venue standards or reviewer minimums, and the one claim asserting a
"cross-dataset + multi-backbone minimum" was **refuted 0-3**. Treat as opinion:

- **First target:** a CVPR/ICCV/ECCV **workshop** (RHOBIN-style — V-VIPE itself
  appeared at CVPR-W) or **BMVC**. Fits an honest, modest-scale, analysis-led
  contribution with a negative result.
- **Journal alternative:** CVIU / Neurocomputing / IEEE TMM if experiments 1–4
  land, giving room for the full requirement-profile argument.
- **Not** a main-conference CVPR/ICCV submission on current evidence: one
  dataset, one backbone, and a substrate that a learned competitor beats.

---

## 7. Risk register

| Risk | Mitigation |
|---|---|
| Reviewer: "3DPCNet already does this, and better" | Lead with the requirement profile (Table 1), not accuracy. Never claim canonicalization novelty. Run experiment 4. |
| Reviewer: "your reliability score is just heuristics without guarantees" | Concede explicitly; add the conformal contrast (experiment 3) and report what label-free costs. |
| Reviewer: "single dataset, single backbone" | Experiment 1 fixes the backbone half. Scope the rest in limitations. |
| "First to…" challenged | Do experiment 6 first; until then claim a requirement profile, not priority. |
| Negative result read as failure | Frame as delimitation with the mechanism (plausibility ≠ depth correctness) and the replication evidence (fusion 4/4, selection straddles chance). |

---

## 8. Paper skeleton

1. **Intro** — frozen predictors, camera-centred outputs, the deployment gap (no training, no labels, no calibration)
2. **Related work** → Table 1 requirement matrix
3. **Method** — analytic body frame (substrate, credited to prior art), multi-scale per-limb extension, the 6-component analytic reliability score
4. **What the signal detects** — degradation sweep (ρ = −0.813, 100% dropout abstention, 0% false abstention)
5. **What it enables** — calibration-free fusion, 4 conditions, 2 subjects, 2 motion regimes
6. **What it does not do** — the delimitation, with within-frame ρ, rank-vs-chance, and the withdrawn selection claim shown in full
7. **Limitations & no-guarantee statement**
8. **Conclusion** — requirement profile as the contribution

Every number cited to its JSON artifact; `evaluation/audit_numbers.py`
(29 claims) keeps the manuscript and the evidence in sync.
