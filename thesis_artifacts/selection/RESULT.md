# Result: the routing experiment — Reading 1, both backbones

Run 2026-08-07, after `PREREGISTRATION.md` was committed (`5dbc47a`). Artifacts:
`selection.json` (MotionAGFormer-XS), `selection_motionbert.json` (MotionBERT).

## The rule (fixed, pre-registered, no tuning)

Route to the **anatomical frame** iff `min(core confidence) ≥ 0.7` **and**
`min(distal confidence) < 0.7`; else route to **template Kabsch**. Confidence
is simulated as `c_j = clip(1 − σ_j/80, 0, 1)` — a detector whose per-joint
confidence drops linearly with localization noise; the oracle column is the
per-pair ceiling `min(anat, t17)` and is labelled as such.

## Distal corruption (from the occlusion protocol)

| σ (mm) | route | routed (XS) | t17 (XS) | routed (MB) | t17 (MB) |
|---|---|---|---|---|---|
| 0 | t17 | 43.30 | 43.30 | 40.96 | 40.96 |
| 20 | t17 | 44.59 | 44.59 | 42.32 | 42.32 |
| 40 | **anat** | 53.45 | 48.05 | **44.13** | 45.98 |
| 80 | **anat** | **53.45** | 59.83 | **44.13** | 58.17 |
| 160 | **anat** | **53.45** | 92.29 | **44.13** | 91.39 |

## Anchor corruption (from the anchor protocol)

| σ (mm) | route | routed (XS) | anat (XS) | routed (MB) | anat (MB) |
|---|---|---|---|---|---|
| 0 | t17 | 43.30 | 53.45 | 40.96 | 44.13 |
| 20 | t17 | 43.70 | 71.95 | 41.38 | 64.57 |
| 40 | t17 | 44.87 | 103.91 | 42.61 | 99.40 |
| 80 | t17 | 49.15 | 215.66 | 47.15 | 217.81 |
| 160 | t17 | 63.22 | 337.87 | 61.63 | 339.86 |

## Verdict

- **R1 (never worse):** worst routed-minus-best over all 20 (regime, σ,
  backbone) cells is **+5.40 mm** (XS, distal σ = 40 — the single transition
  cell); tolerance 7 mm. MotionBERT is 0.00 mm at every cell. **Holds.**
- **R2 (strict gain):** at σ = 160 distal the routed arm beats template-only,
  cluster-bootstrap CI excluding zero on both backbones:
  XS [+30.2, +45.3] (mean −38.8 mm), MB [+44.8, +49.8] (mean −47.3 mm).
  **Holds.**
- **R3 (never routes into the collapse):** under anchor corruption σ ≥ 40 the
  rule routes to template at every cell. **Holds.**
- **R4 (clean):** at σ = 0 the rule routes to template — the better method at
  clean — on both regimes and backbones. **Holds.**

**Reading 1: routing established.** The routed distance attains the better
single alignment's aggregate mean at 19 of 20 cells, trails it by 5.4 mm in the
one transition cell (where the two methods' bootstrap intervals overlap and
their crossover severity differs between backbones), and is 38–47 mm better
than template-Kabsch alone under severe distal corruption. The routed distance
approaches the per-pair oracle where the methods are separated (e.g. XS distal
σ=160: 53.45 routed vs 49.16 oracle) and is 7.9 mm from it in the single
transition cell (σ=40 XS), which is the same cell where it trails the better
single alignment.

## What this answers

The report's §5.6.1 question — *"then why keep the anatomical frame at all?"* —
now has an experimental answer, not only the rhetorical one (the baseline has
no anatomical axis to hold fixed and vary). The frame is the right arm exactly
when the core joints are reliable and the periphery is not; the rule is never
worse than the best single alignment within the stated tolerance and strictly
better under severe distal corruption. This is the practical payload of the
failure-surface map in `thesis_artifacts/anchor_corruption/RESULT.md`.

## Honest boundaries

- The confidence signal is **simulated, not measured, and noiseless**: the
  corruption experiments inject noise with no confidence channel, and the
  signal is an exact deterministic function of (severity, corrupted set). It is
  therefore an **upper bound** on what a real, noisy detector-confidence
  channel could provide, and it is labelled as such in the code, the JSON and
  the report. In deployment the rule consumes the detector's real per-joint
  confidence; the correspondence between injected noise and detector
  confidence is a model.
- The evaluation is of the **conditional choice per corruption level**, not of
  per-frame routing: confidence is constant within a level, so the decision is
  constant within a level. Per-frame deployment driven by noisy, per-frame
  confidence is the rule's assumption, not something this experiment measures.
- "Never worse" holds for **aggregate means** within a pre-registered 7 mm
  transition allowance, on one dataset, two backbones, under Gaussian 3D
  corruption. Pair by pair it does not hold in the transition cell (XS distal
  σ = 40): the routed arm (anatomical, 53.45 mm) is worse than template
  (48.05 mm) for every pair there, and routed minus the per-pair oracle is
  7.9 mm in that cell. The 5.4 mm shortfall is a point estimate whose interval
  is not established — at the neighbouring severity (σ = 80) the XS interval
  spans zero.
- Not a claim of optimality: the threshold is fixed, not tuned, and was
  committed before this experiment ran. The rule does not make the anatomical
  frame win; it makes the *combination* no worse, in aggregate mean, than the
  better single alignment. It is never *better* than the better single arm --- a
  selector cannot exceed the arm it selects. Routed equals best-single in 9 of
  10 cells and is 5.4 mm worse in the tenth. The 38-47 mm margin is over
  template alignment ALONE, not over the better arm.

## Effect on the frozen report

None — the report's 255 audited claims and nine pre-registrations are
unchanged. This experiment is cited by the minimal report and the defence prep.
