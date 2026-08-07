# Pre-registration: anchor-joint corruption (the symmetric half of the failure map)

Committed **before** this experiment runs, per the repository's standing
pre-registration discipline. Run date: 2026-08-07. Defence: 2026-08-09.

## Question

The distal-corruption experiment (`thesis_artifacts/occlusion/`) found the
anatomical frame stays exactly flat while the template-Kabsch baseline
degrades, because the frame reads only four joints and none of the corrupted
eight are among them. The mirror question was never asked: the frame's error
must be concentrated in that four-joint support. This experiment corrupts the
support joints and measures the symmetric surface.

## Protocol

- **Backbones:** MotionAGFormer-XS (`h36m_replication/preds.npz`) and
  MotionBERT (`h36m_motionbert/preds_motionbert.npz`), both frozen, predictions
  cached. No inference runs.
- **Corrupted joints:** `{1, 4, 8}` — right hip, left hip, thorax — the frame's
  support minus the root (the root is re-centred away before corruption, as in
  the distal experiment, so corrupting it would add only a common translation
  both alignments ignore).
- **Scored joints:** `{9, 10, 11, 14}` — neck, head, both shoulders —
  uncorrupted and non-constructor, **identical to the distal experiment**, so
  the σ = 0 row is an identity control that must reproduce the distal run
  exactly and the two tables read side by side.
- **Severities:** σ ∈ {0, 20, 40, 80, 160} mm isotropic Gaussian, per-camera
  independent seeds that are pure functions of (group, camera, severity).
- **Arms:** A anatomical frame; B template-Kabsch fitted on all 17 joints;
  C template-Kabsch fitted on the constructor set `{0,1,4,8}` (the control that
  separates *support* from *construction*).
- **Metric:** cross-view joint distance on the scored joints, mean over 180
  held-out camera pairs, cluster-bootstrapped over 30 subject-action groups.

## Criteria

- **C1 (sanity, identity control):** at σ = 0 the verdict is
  `template_better` — reproducing the distal experiment's σ = 0 row
  (anat 53.45 / t17 43.30 / t4 49.69 on XS; 44.13 / 40.96 / 47.05 on MB).
- **C2 (claim):** at σ ∈ {20, 40} the anatomical frame is **worse** than
  template17, with the 95 % cluster-bootstrap CI excluding zero, **on both
  backbones**. One backbone is not two — the rule by which the conditioning
  and occlusion results were rejected applies here too.
- **C3 (support control):** arm C (four-joint fit on the corrupted set) is not
  meaningfully better than the anatomical frame at σ ∈ {20, 40} on both
  backbones — the collapse is a property of the four-joint support, not of the
  anatomical construction.
- **C4 (monotonicity):** the anatomical frame's mean distance strictly
  increases with σ on both backbones — its error is a pure function of its
  support joints.

## Readings

1. C1 ∧ C2 ∧ C3 ∧ C4: **support-concentrated failure established.** The two
   alignments have disjoint failure supports: distal corruption is invisible to
   the frame and visible to the template; anchor corruption collapses the frame
   and is diluted in the 17-joint fit. This completes the map that the routing
   rule in `evaluation/selection_rule` exploits.
2. C2 on one backbone only: failure — same rule as the tenth pre-registration.
3. Otherwise: not established.

## Claims this will and will not support

Will support (if reading 1): *"the anatomical frame's cross-view error is a
function of its four constructor joints alone; corrupt them and it collapses,
while the 17-joint Kabsch fit degrades gracefully."*

Will **not** support: any claim that the anatomical frame is more accurate, or
that this constitutes a regime where the anatomical frame *wins*. This is the
failure half of the map; the selection experiment is where a routing rule turns
the complementarity into a never-worse-than-the-best result.

## Effect on the frozen report

None. The report's 255 audited claims, its nine pre-registrations and its
section structure are unchanged. This is a post-freeze experiment recorded in
`thesis_artifacts/anchor_corruption/`, cited by the minimal report and the
defence prep only.
