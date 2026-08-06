# Literature sweep, 7 Aug 2026 — what it found, and the one thing you must absorb

104 agents, 25 claims put to 3-vote adversarial verification, 9 refuted. Read
this before the defence. It contains one liability, one usable gap, and three
things that are **not** evidence and must not be spoken as if they were.

---

## 1. THE LIABILITY: your frame is already published as a losing baseline

**3DPCNet (arXiv 2509.23455) is peer-reviewed — ICASSP 2026, not a preprint —
and its Section 3.3 "Geometric Baseline" is a two-vector TRIAD-family
construction of your family.** Verbatim from the paper:

> "This approach does not require learned parameters and is based on anatomical
> landmarks. It first defines a plane using the shoulder and hip joints and
> rotates the pose to align this plane's normal vector with a fixed axis (e.g.,
> the Z-axis). A second rotation is then applied to align the vector between the
> shoulders parallel to the X-axis."

Their Table 1:

| | MPJPE | Rotation error |
|---|---|---|
| GEOMETRIC (their anatomical baseline), MM-Fi S2 | 62.85 mm | **20.64°** |
| 3DPCNet | 47.57 mm | **3.58°** |
| GEOMETRIC, MM-Fi S3 | 64.58 mm | 21.56° |
| 3DPCNet | 47.71 mm | 4.24° |

An examiner who reads your related-work section and then opens 3DPCNet finds
your method family published losing by ~6× on rotation error. **Assume they
will.** You need the answer below ready, and you should raise it yourself.

### Four qualifications that are real, verified, and yours to use

1. **The MPJPE column is confounded and the authors concede it.** 3DPCNet makes
   non-rigid shape edits the geometric frame never attempts. Only the rotation
   column is a clean canonicalization comparison.
2. **It is the proposing authors' own self-implemented baseline** — one table, no
   error bars, no seeds, no oracle upper bound. 20° is high enough for a
   deterministic, exactly-determined frame to suggest a target-definition
   mismatch or noisy MM-Fi ground truth rather than an intrinsic ceiling.
3. **It measures a different thing than you do.** Theirs is anatomical-vs-learned
   against a *ground-truth canonical pose*. Yours is prediction-vs-prediction
   agreement across cameras. Their result corroborates yours by analogy only.
4. **Their primary axis differs from yours.** They take the shoulder/hip plane
   *normal* as primary; your frame does not. Say **"TRIAD-family"**, never
   "identical".

### What you say

> Sir, the closest prior work publishes a baseline of my family and reports it
> losing to their learned module by six times on rotation error. I should raise
> that rather than wait for it. Three things qualify it: it is their own
> self-implemented baseline with no error bars, it measures alignment to a
> ground-truth canonical pose rather than agreement between two cameras, and its
> primary axis is not mine. But the direction agrees with my own finding, and I
> would rather say that the evidence points the same way twice than argue it
> away.

---

## 2. The one defence that does NOT work — know this before you use it

Rigid maps preserve shape exactly: the geometric baseline scores **PA-MPJPE =
0.00 mm** by construction, while 3DPCNet's learned residual introduces ~37.5 mm
of non-rigid deformation, which its own authors list as a limitation and scope a
fix for.

**This is tempting and it does not save you.** Exact rigidity is a property of
*any* rigid map — including the Kabsch-to-template baseline that beat you
180/180. It separates rigid from learned. It does not separate anatomical from
Kabsch. Do not deploy it as a defence of your frame specifically.

---

## 3. The verdict on novelty: no geometric novelty is available

**As of this sweep there is no confirmed axis on which the anatomical frame
beats Kabsch-to-template.** Not one. Combined with your two failed
pre-registrations, the honest position is that the defensible contribution is
the **pre-registered boundary delimitation itself**, plus the demographic gap
below — not any claim of method superiority.

### The one live gap: atypical morphology

3DPCNet evaluates on exactly two able-bodied corpora — MM-Fi (40 subjects) and
TotalCapture (5 subjects) — and its own future-work item (iv) is to *"validate
clinical and performance metrics end-to-end (e.g., ROM, asymmetry indices) in
real deployments."*

This is your best future-work sentence, and it is structurally where a
template-free frame should win: Kabsch-to-template needs a skeleton resembling
the subject; yours needs only that the subject has hips and a torso.

**Two phrasing traps.** "Able-bodied" comes from the MM-Fi paper, not from
3DPCNet, which says only "40 subjects" — attribute it correctly. And
TotalCapture contains an activity literally labelled "ROM", so say *"3DPCNet
defers ROM and asymmetry-index validation to future work, by its own
admission"*, never *"3DPCNet never mentions ROM"*.

**Confidence: medium.** This is one paper's omission, not a field survey. Say
"I found no prior work evaluating this", never "no prior work exists".

---

## 4. Three things that are NOT evidence — do not speak them as findings

| | Status |
|---|---|
| **Asymmetric failure modes** (unilateral occlusion, hemiplegia, amputee gait) favouring anatomical frames | **Zero surviving claims.** Nothing found in vision or biomechanics. Theoretically live, entirely unsupported. Two prior searches in this family already failed, so the prior is poor |
| **Cross-view / prediction-vs-prediction agreement as a named evaluation axis** | **Refuted 1-2 and 0-3.** You currently **cannot** assert this axis is unclaimed, name who uses it, or say what it is called. This sits directly beneath your central framing |
| **No CV or biomechanics negative-results venue exists** | **Unproven.** No systematic sweep of CVPR/ICCV/ECCV/WACV CFPs or biomechanics journal policies was completed |

The Q4 hole is the most uncomfortable one. If an examiner asks *"is cross-view
consistency an accepted evaluation axis, and what do others call it?"*, the
honest answer today is: **"I measure it, I define it precisely in my protocol
section, and I have not established that it is a named standard in the
literature."** That is a survivable answer. Inventing a citation is not.

---

## 5. The theoretical lead — read it, cite it, do not build on it

**Dym, Lawrence & Siegel, "Equivariant Frames and the Impossibility of
Continuous Canonicalization", ICML 2024 (arXiv 2402.16077).** Verified: real
paper, real authors, and it explicitly treats **SO(3) acting on point clouds**.

Its result: for commonly-used groups there is no efficiently computable frame
choice that preserves continuity, and unweighted frame-averaging can turn a
smooth function into a discontinuous one. It proposes **weighted frames** as the
fix.

**Why this matters to you.** If it applies, it reframes your thesis from *"my
frame lost"* to *"no deterministic training-free frame can be continuous
everywhere, and this thesis maps where each one breaks."* That is a principled
boundary contribution rather than a failed hypothesis, and it would explain your
degeneracy gates as a necessity rather than an engineering patch.

**Why you must not lean on it on Saturday.** The claim failed adversarial
verification 1-2, and its testable corollary failed 0-3. Your canonicalization
is a single deterministic frame, not frame-averaging over an orbit — the
neighbourhood is right, the identity is not established. Spend 30 minutes
reading the actual theorem's hypotheses. **Cite it as related work that "suggests
a principled reason such constructions have degenerate configurations." Do not
claim the theorem covers your construction unless you have read and checked it.**

---

## 6. Publication venues: yes in principle, no in practice for this cycle

Every confirmed negative-results venue is discipline-mismatched, gated, or
closed. **No CV or biomechanics track was found.**

| Venue | Why it does not work |
|---|---|
| MLRC @ NeurIPS 2026 | Requires TMLR acceptance by 30 Sep 2026; TMLR median decision 76–91 days. Unreachable |
| ReScience C | Only rolling option, but scope is replication of a *named published* claim; self-replication barred; 16-week median |
| ICLR 2026 Blogposts | Closed 7 Dec 2025. No 2027 CFP published |
| ISSRE RENE | Software engineering. Closed 12 Jul 2026 |
| ICBINB 2026 | LLM-scoped; closed 31 Jan 2026 |
| JASNH | Psychology only, and requires the null to be *supported* — Kabsch winning 180/180 is a decisive directional finding, not a null |

Both ReScience C and MLRC define "negative result" as failure to reproduce a
*published* claim. Yours is an original disconfirmation of your own
pre-registered hypothesis — out of scope at both. The only identified route is
reframing as an explicit replication of V-VIPE's Kabsch body-frame
preprocessing, which needs new work against their reported numbers. **Not before
Saturday.**

---

## What to actually do

1. **Add the 3DPCNet geometric-baseline answer to your prep.** 20 min. This is
   the only item that changes your defence.
2. **Read arXiv 2402.16077's theorem statement.** 30 min. Cite as related work
   only.
3. **Prepare the Q4 answer** — "I define it in my protocol; I have not
   established it is a named standard." 10 min.
4. **Use atypical morphology as your future-work answer**, phrased as "I found no
   prior work", not "none exists". 10 min.
5. **Run no further experiments.** Three searches for a winning regime have now
   failed, and the literature says the geometric ground is occupied.
