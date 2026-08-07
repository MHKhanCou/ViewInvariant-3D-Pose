# Result: the anchor-corruption experiment — Reading 1, both backbones

Run 2026-08-07, after `PREREGISTRATION.md` was committed (`5dbc47a`). Artifacts:
`anchor_corruption.json` (MotionAGFormer-XS), `anchor_corruption_motionbert.json`
(MotionBERT). The σ = 0 row reproduces the distal experiment's identity control
exactly (XS 53.45 / 43.30 / 49.69; MB 44.13 / 40.96 / 47.05), so the two tables
are directly comparable.

## Reading: support-concentrated failure established

Corrupt the frame's support joints {1 r_hip, 4 l_hip, 8 thorax} and it
collapses at every severity, on both backbones, monotonically, with the
four-joint control (arm C, Kabsch fitted on the same corrupted set) collapsing
with it. The 17-joint Kabsch fit degrades gracefully.

| σ (mm) | XS: anat / t17 / t4 | XS verdict | MB: anat / t17 / t4 | MB verdict |
|---|---|---|---|---|
| 0 | 53.45 / 43.30 / 49.69 | template better | 44.13 / 40.96 / 47.05 | template better |
| 20 | 71.95 / 43.70 / 64.24 | template better | 64.57 / 41.38 / 62.06 | template better |
| 40 | 103.91 / 44.87 / 93.43 | template better | 99.40 / 42.61 / 91.96 | template better |
| 80 | 215.66 / 49.15 / 164.18 | template better | 217.81 / 47.15 / 163.55 | template better |
| 160 | 337.87 / 63.22 / 304.95 | template better | 339.86 / 61.63 / 305.42 | template better |

Criterion C2 (anatomical worse than template17 at σ ∈ {20, 40}, CI excluding
zero, both backbones) holds: the worst CI is [−37.4, −21.1] (XS σ = 20) and
every interval excludes zero. C3 (the four-joint control fails with the frame)
holds: arm C tracks the collapse, 304.95–305.42 mm at σ = 160, proving the
collapse is a property of the four-joint support, not of the anatomical
construction. C4 (monotone collapse) holds: 53.45 → 71.95 → 103.91 → 215.66 →
337.87 (XS) and 44.13 → 64.57 → 99.40 → 217.81 → 339.86 (MB).

## What this means, read with the distal result

| Corruption | Anatomical frame | Template-Kabsch (17 joints) |
|---|---|---|
| distal joints (occlusion/) | **flat** — 53.45 / 44.13 mm at every σ, it never reads them | degrades — 43.30 → 92.29 (XS) |
| anchor joints {1,4,8} (this run) | **collapses** — 53.45 → 337.87 (XS) | graceful — 43.30 → 63.22 (XS) |

The two alignments have **disjoint failure supports**: distal corruption is
invisible to the frame and visible to the template; anchor corruption is
catastrophic for the frame and diluted in the 17-joint fit. Neither alignment
is better than the other in general — they fail on different joints. The
routing rule in `thesis_artifacts/selection/RESULT.md` exploits exactly this.

## Honest boundaries

- The corruption is injected Gaussian noise on predicted 3D joints, not real
  occlusion or detector failure. Real corruption is future work, exactly as the
  occlusion pre-registration said.
- This is the *failure* half of the map. It establishes nothing about which
  alignment wins at clean data (the template does) and nothing about the
  anatomical frame's accuracy.

## Effect on the frozen report

None — the report's 255 audited claims and nine pre-registrations are
unchanged. This is the twelfth pre-registration overall, run after the freeze.
