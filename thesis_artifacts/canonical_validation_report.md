# Canonical Pose Validation Report

## Summary

This report documents the mathematical validation of the canonical body-frame pose representation for the thesis "View-Invariant 3D Human Pose Estimation from RGB Images."

## Baseline Results

| Metric | Official Paper | Reproduced |
|--------|---------------|------------|
| MPJPE | 45.1 mm | **45.149 mm** |
| P-MPJPE | 36.9 mm | **36.892 mm** |

The reproduction matches the official paper within 0.1 mm, confirming correct preprocessing, checkpoint loading, architecture, and evaluation pipeline.

## Canonical Module Validation

### Mathematical Validation (Synthetic Tests)

The canonical pose representation is validated through synthetic rigid-rotation invariance tests:

| Test | Result |
|------|--------|
| Number of random rotations tested | 100 |
| Maximum raw synthetic-coordinate error | 1.19e-07 |
| Maximum relative error (error / pose extent) | 5.96e-08 (dimensionless) |
| Mean canonical pose error | 8.02e-08 mm |
| Orthonormality error (R^T R - I) | 0.0 |
| Root joint is zero after canonicalization | True |
| Zero pose returns zero output | True |
| Zero pose returns identity R | True |
| Zero pose marked as invalid | True |
| Bone lengths all positive | True (16 bones) |

**Interpretation:** The canonical representation is mathematically correct. Rigid rotations of the input produce identical canonical outputs (within floating-point precision). The rotation matrix is orthonormal. Degenerate inputs are handled gracefully.

### Cross-View Evaluation Status

Exact cross-view correspondence and metric 3D ground-truth semantics could not be verified from the available preprocessed pkl fields. The following limitations were identified:

1. **No temporal timestamps**: Source strings (e.g., `s_09_act_02_subact_01_ca_01`) lack frame-level temporal identifiers suitable for cross-camera pairing.

2. **Camera identity not uniquely determined by source**: Source strings do not uniquely identify cameras in this preprocessed pkl, preventing reliable sequence-to-camera mapping.

3. **GT coordinate semantics unverifiable**: `joints_2.5d_image` has coordinate semantics that could not be verified as metric 3D from the available pkl fields. GT canonical consistency floor is not computable.

Note: Human3.6M in its raw form may support cross-view evaluation with different preprocessing. The limitation identified here is specific to this preprocessed pkl's metadata fields.

These limitations are detected by `canonical/metadata_capability.py` and verified by `canonical/test_evaluator.py` (18 tests, all passing).

## Qualitative Web Demo

The canonical mode is available in the Gradio web application:

- **Camera-relative root pose**: Zeroes root joint, renders with fixed viewing angle
- **Canonical body-frame pose**: Constructs body-fixed coordinate system, renders with equal axis scaling

Both modes are qualitative visualizations for thesis demonstration purposes. They do not represent metric 3D reconstruction accuracy.

## Limitations

1. **Mathematical validation only**: The canonical module is validated through synthetic rigid-rotation tests, not real cross-view dataset evaluation.

2. **No cross-view metrics**: Exact cross-view correspondence and metric 3D ground-truth semantics could not be verified from the available preprocessed pkl fields.

3. **Detector domain shift**: The demo uses YOLOv8-pose on in-the-wild images, which differs from the Stacked-Hourglass detections used during training.

4. **Monocular depth ambiguity**: Canonicalization normalizes orientation but cannot resolve depth ambiguity from single-view input.

5. **Single-person only**: The demo selects the highest-confidence person per frame.

## Files

| File | Description |
|------|-------------|
| `baseline_results.json` | Official and reproduced baseline metrics |
| `canonical_validation.json` | Synthetic test results |
| `canonical_validation_report.md` | This report |
| `figures/` | Qualitative visualization outputs |

## Conclusion

The canonical body-frame representation is mathematically validated as a deterministic geometric post-processing step. It normalizes camera-orientation variation when body axes are reliably estimated. Cross-view quantitative evaluation is not possible with the current processed H3.6M metadata, but the mathematical correctness of the canonicalization is confirmed through synthetic tests.
