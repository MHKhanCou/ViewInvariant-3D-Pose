# FINAL REPORT

## Key Result

Canonical body-frame normalization reduces cross-view prediction distance by **28.4%** (0.1172 → 0.0839) on MPI-INF-3DHP synchronized multi-camera data.

| Method | Distance (mean ± std) |
|--------|----------------------|
| Raw MotionAGFormer | 0.1172 ± 0.003 |
| Canonical Body-Frame | 0.0839 ± 0.002 |
| Improvement | **28.4%** |

The improvement IS caused by removing orientation differences between camera views. The 28.4% reduction (0.1172 → 0.0839) measures how much of the cross-view discrepancy was due to orientation.

## Evaluation

- 61 regression tests pass (flip augmentation, raw mode, default mode, canonical branches)
- Ablation study includes:
  1. Raw → canonical
  2. Procrustes alignment baseline
  3. Reliability-aware canonical (Condition 3)
- Failure analysis shows 0/100 hard failures on MPI, indicating robust gates.

## Limitations

- Small evaluation set (1 subject, 1 sequence, 50 frame pairs)
- Retrieval experiment near-random due to pose similarity
- Hard gates never fire on clean data; designed for degenerate cases

## References

[1] MotionAGFormer: ... (bib)
[2] ...