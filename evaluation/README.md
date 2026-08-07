# evaluation/

Every module writes a JSON artifact under `thesis_artifacts/` and is runnable on
its own. `audit_numbers.py` re-derives 304 headline claims from those artifacts
and fails on drift, so the artifacts are the source of truth and the report
quotes them rather than restating them.

Run anything with `./venv/Scripts/python.exe -m evaluation.<module>`.

## Order of execution

Later stages read caches written by earlier ones. From a clean checkout:

```
run_eval            MPI-INF-3DHP predictions -> per-camera cache   (slow)
h36m_replication    Human3.6M predictions    -> preds.npz          (~50 min)
```

Everything else reads one of those two caches and takes seconds to minutes.

## MPI-INF-3DHP

| Module | What it produces |
|---|---|
| `protocol.py` | camera discovery, pairing table, frame-id parsing |
| `run_eval.py` | prediction cache + cross-view distances (the main run) |
| `lifting.py` | shared 2D->3D path, identical to the demo app |
| `metrics.py` | cross-view joint distance, bone deviation, joint angles |
| `oracle.py` | per-frame Procrustes floor |
| `gt_eval.py` | ground-truth anchoring, similarity-aligned error |
| `multiscale_eval.py` | per-limb frames vs the global frame |
| `fusion.py` / `fusion_eval.py` | calibration-free fusion and view selection |
| `reliability.py` | the six-component score (falsified; see report) |
| `corruptions.py` / `degradation_sweep.py` / `degradation_analysis.py` | controlled degradation |
| `ablation.py` | abstention and coverage-error |
| `retrieval.py` | cross-view retrieval (negative result) |
| `backbone_transfer.py` | same module over three predictors |
| `bone_consistency.py` | temporal bone-length signal (retracted; see below) |
| `tta_consistency.py` | test-time augmentation dispersion (pre-registered, failed) |

## Human3.6M

Added for the cross-dataset replication. All four read the same `preds.npz`, so
only `h36m_replication --stage infer` costs real time.

| Module | What it produces |
|---|---|
| `h36m_replication.py` | predictions via the official eval path; bone-length replication |
| `h36m_crossview.py` | cross-view canonicalization, 180 held-out pairs |
| `h36m_multiscale.py` | per-limb frames, plus the bilateral-asymmetry variant |
| `h36m_fusion.py` | fusion over four uncalibrated views |

## Reporting

| Module | What it produces |
|---|---|
| `audit_numbers.py` | re-derives 304 claims from artifacts; fails on drift |
| `make_figures.py` | MPI-INF-3DHP figures |
| `make_h36m_figures.py` | Human3.6M figures |
| `make_appendix_tables.py` | Appendix A of the report, generated not typed |
| `report.py` | summary text |

## Two things not to undo

`canonical/multiscale.py` and `evaluation/fusion.py` are deliberately
unmodified even though both have documented defects. The MPI-INF-3DHP numbers
in the report were produced with them and are audited, so corrections are
implemented as variants in the Human3.6M modules and reported alongside the
originals:

- `h36m_multiscale.py` carries `SEGMENTS_SYMMETRIC`, which fixes a bilateral
  asymmetry in the leg axis definitions. Only the left leg changes.
- `h36m_fusion.py` carries `resolve_reflections_anatomical`, which was tested
  and made results **worse**. It is retained as a measured control, not as a fix.

The bone-length signal in `bone_consistency.py` scored +0.492 on MPI-INF-3DHP
and +0.098 on Human3.6M, failing all five criteria. **Do not quote +0.492 on its
own**; the report retracts the general claim.
