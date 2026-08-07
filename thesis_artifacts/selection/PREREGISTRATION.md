# Pre-registration: confidence-gated routing between the two alignments

Committed **before** this experiment runs, per the repository's standing
pre-registration discipline. Run date: 2026-08-07. Defence: 2026-08-09.

## Question

Section 5.6.1 establishes that template Kabsch beats the anatomical frame on
the headline metric. The two post-freeze corruption experiments establish why
they are different instruments, not competitors: the anatomical frame reads
four joints and is exactly flat under distal corruption while Kabsch degrades
(occlusion/); the frame collapses the moment those four joints are corrupted
while Kabsch degrades gracefully (anchor_corruption/). Question: can a fixed,
training-free decision rule that routes each frame to one of the two alignments
be **never worse than the better single alignment**, and strictly better than
template-Kabsch alone where the methods are separated?

## Rule (fixed, no tuning on the data)

Route to the **anatomical frame** iff

    min over {1 r_hip, 4 l_hip, 8 thorax} of c_j  >=  0.7     (core reliable)
    AND
    min over the eight distal joints of c_j       <  0.7     (periphery broken)

else route to **template Kabsch**. `c_j` is a per-joint confidence; the
threshold 0.7 means the relevant joints' confidence has dropped to 70 %.

## Input signal

The corruption experiments inject noise with no confidence channel, so `c_j` is
**simulated** as a calibrated function of the injected noise:
`c_j = clip(1 - sigma_j / 80, 0, 1)` where `sigma_j` is the noise applied to
joint j (0 for clean joints). This models a detector whose per-keypoint
confidence drops linearly with localization error and saturates at 80 mm of
noise. The simulation is stated in the code, the output and the report; it is
not presented as measured confidence. In deployment the same rule consumes the
detector's real per-joint confidence (e.g. YOLOv8 keypoint scores).

## Protocol

- **Backbones:** MotionAGFormer-XS and MotionBERT, frozen, cached predictions.
- **Regimes:** distal corruption (occlusion protocol, corrupted `{2,3,5,6,12,13,15,16}`)
  and anchor corruption (anchor protocol, corrupted `{1,4,8}`).
- **Severities:** σ ∈ {0, 20, 40, 80, 160} mm.
- **Per pair:** anatomical distance, template17 distance, routed distance (the
  chosen arm), oracle distance (`min(anat, t17)` — the ceiling, labelled as
  oracle).
- **Bootstrap:** cluster over 30 subject-action groups, as throughout.

## Criteria

- **R1 (never-worse):** routed mean minus the better single alignment's mean
  is ≥ −7 mm at **every** (regime, severity) cell, both backbones. The 7 mm
  allowance covers the transition band where the two methods' bootstrap
  intervals overlap and the crossover severity differs between backbones
  (MotionBERT crosses at σ = 40, MotionAGFormer-XS between 80 and 160).
- **R2 (strict gain where separated):** at σ = 160 distal, routed (the
  anatomical arm) beats template17 with the cluster-bootstrap CI excluding
  zero, both backbones.
- **R3 (never routes into the collapse):** under anchor corruption at
  σ ≥ 40 the rule routes to template (never into the collapsed arm).
- **R4 (clean):** at σ = 0 both regimes, the rule routes to template — the
  better method at clean.

## Readings

1. R1 ∧ R2 ∧ R3 ∧ R4: **routing established.** The failure-surface map yields a
   decision rule that is never worse than the best single alignment (within the
   stated 7 mm transition allowance) and strictly better under severe distal
   corruption. This is the answer to "then why keep the anatomical frame at
   all": it is the right arm exactly when the periphery is broken and the core
   is not.
2. Otherwise: routing not established; report the table descriptively.

## Claims this will and will not support

Will support (if reading 1): *"the two training-free alignments have disjoint
failure supports, and a confidence-gated routing rule attains the better
alignment's cross-view distance at every severity tested, within 7 mm in the
transition band, and beats template-Kabsch alone by 38–47 mm under severe
distal corruption."*

Will **not** support: any claim that the anatomical frame is more accurate, or
that the rule's thresholds are optimal (they are fixed, not tuned; tuning on
the test split would be the failure mode the pre-registrations exist to
prevent).

## Effect on the frozen report

None. The report's 255 audited claims and nine pre-registrations are
unchanged. This is a post-freeze experiment, cited by the minimal report and
the defence prep only.

*Updated 8 Aug 2026: the report has since been extended to report this experiment, and the audit now stands at 304 claims across seventeen pre-registered experiments. The statement above describes the state at the time this pre-registration was committed and is left unedited for that reason.*
