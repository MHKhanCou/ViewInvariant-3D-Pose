# View-Invariant 3D Human Pose Estimation

**Undergraduate thesis — Computer Science, Comilla University**

Making the output of a *frozen* monocular 3D pose estimator comparable across camera viewpoints, with **no training, no labels, no camera calibration, and zero added parameters.**

![teaser](thesis_report/images/fig_teaser.png)

---

## What this is

Monocular 3D pose estimators return skeletons in the observing camera's coordinate frame, so two cameras watching the same person at the same instant produce two different sets of numbers for the same pose. This project builds a body-fixed frame from the predicted anatomy and applies it *after* prediction, which cancels the camera rotation exactly.

The frame construction is not new — it is the **TRIAD** algorithm from spacecraft attitude determination (Black, 1964), and the rule that the better-determined axis should be primary is due to Shuster & Oh (1981). The propagation of landmark error into frame orientation is established in biomechanics (Della Croce et al., 1999, 2005). **None of that is claimed here.**

What the project contributes is an experimental answer to a narrower question:

> **What determines whether an anatomical reference frame is consistent across viewpoints — and where does that reasoning stop applying?**

The answer is a boundary, established by three pre-registered tests, two of which failed.

---

## Headline results

Human3.6M, 180 held-out camera pairs, subjects S9 and S11, which took no part in developing the method.

| Result | Value |
|---|---|
| Cross-view distance reduction | **74.1 %** (CI [+69.8, +77.2], clustered by subject-action) |
| Pairs improved | **179 / 180** |
| Gap closed to per-frame Procrustes oracle | **90.5 %** |
| Second backbone (MotionBERT, 19× larger, unmodified code) | **77.5 %**, 180 / 180 |
| Trained parameters added | **0** |
| Cost per frame | **402 FLOPs** (0.0005 % of the backbone) |

All improvement figures are the mean over camera pairs of each pair's own percentage, not the ratio of aggregate means. That convention is the conservative one and is stated in the report; dividing the table entries will not reproduce it.

### The negative results, which are half the thesis

| Claim tested | Outcome |
|---|---|
| Axis length decides *between* frame constructions | **Holds** — ρ = +0.904 / +0.880 on two backbones |
| …decides *which frame* within a construction to trust | **Fails** — indistinguishable from a random null |
| …decides *which joint* will disagree | **Fails** — articulation dominates; torso-rigid joints sit at a *larger* radius yet disagree **2.5× less** |
| Temporal bone-length inconsistency predicts error | **Retracted** — ρ = +0.492 on one dataset, +0.098 on the other |
| Test-time augmentation dispersion predicts error | **Fails** all three pre-registered criteria |
| The analytic reliability score predicts accuracy | **Falsified** five independent ways |
| …but does it gate *canonicalization quality*? | **Yes**, on both backbones (reported as exploratory) |

Four pre-registrations were committed to version history **before** the experiments they govern. Two of them the results then contradicted.

---

## Reproducing the claims

Every number in the 87-page report is recomputed from stored artifacts:

```bash
python -m evaluation.audit_numbers      # 167 claims verified against thesis_artifacts/
python -m unittest discover -s tests -q # 72 tests
python -m presentation.render --teaser  # regenerate every figure from the artifacts
```

`audit_numbers.py` fails loudly if any reported figure drifts from the JSON artifact it came from. The figures are generated from the same artifacts, so a figure cannot disagree with the number it illustrates.

---

## Repository map

```
canonical/        body-frame construction, multi-scale and multi-landmark variants
evaluation/       one module per experiment, each writing a JSON artifact
  audit_numbers.py        recomputes all 167 reported claims
  h36m_crossview.py       the central cross-view result
  axis_length_law.py      the quantitative fit and its bootstrap
  conditioning_abstention.py  pre-registered abstention test (failed)
  radial_law.py           pre-registered joint-level test (failed)
presentation/
  render.py         all report figures + the two-view comparison, from artifacts
  bvh_export.py     body-relative BVH export
thesis_artifacts/ stored results + the four PREREGISTRATION.md files
thesis_report/    the report, DEFENSE_QA.md, FREEZE_CHECKLIST.md
tests/            72 tests, no dataset required
```

---

## Setup

Python 3.11, PyTorch (CPU is sufficient for the demo).

```bash
git clone https://github.com/MHKhanCoU/ViewInvariant-3D-Pose.git
cd ViewInvariant-3D-Pose
python -m venv venv && .\venv\Scripts\activate   # Windows
pip install -r requirements.txt
pip install ultralytics gradio
```

**Dataset** — download [MotionBERT's preprocessed Human3.6M](https://1drv.ms/u/s!AvAdh0LSjEOlgU7BuUZcyafu8kzc?e=vobkjZ), unzip to `data/motion3d/`, then:

```bash
cd data/preprocess && python h36m.py --n-frames 27
```

**Checkpoint** — [MotionAGFormer-XS](https://drive.google.com/file/d/1Pab7cPvnWG8NOVd0nnL1iqAfYCUY4hDH/view?usp=sharing) into `checkpoint/`.

## Usage

```bash
python app.py                                        # Gradio demo
python demo_live/infer_image.py --input image.jpg    # CLI, image
python demo_live/infer_video.py --input video.mp4    # CLI, video
python train.py --eval-only --checkpoint checkpoint \
    --checkpoint-file motionagformer-xs-h36m.pth.tr \
    --config configs/h36m/MotionAGFormer-xsmall.yaml # baseline: 45.149 mm MPJPE
```

The baseline reproduces the published MotionAGFormer-XS figure to three decimals (45.149 mm against 45.1 mm reported), which is what makes the failed replications in this project interpretable — a negative result is only informative if the apparatus is not what failed.

---

## Limitations, stated plainly

- **The metric measures agreement, not correctness.** Two predictions that are wrong in the same way agree perfectly. The Procrustes oracle floor and the retained correlation with ground-truth error bound this, but do not eliminate it.
- **No downstream task succeeds.** Cross-view retrieval is negative and used a superseded protocol. The project improves a geometric quantity and does not show that improving it improves an application.
- **Two datasets, both laboratory multi-camera rigs; two backbones, both transformers.** "Model-independent" rests on n = 2.
- Single person per frame; no absolute scale; no world-space trajectory.
- The demo is qualitative. All benchmark numbers come from the evaluation modules, not from the demo path.

---

## Related work

| Work | Reference |
|---|---|
| **MotionAGFormer** — backbone | Mehraban, Adeli & Taati, *MotionAGFormer: Enhancing 3D Human Pose Estimation with a Transformer-GCNFormer Network*, WACV 2024. [arXiv](https://arxiv.org/abs/2310.16288) · [code](https://github.com/TaatiTeam/MotionAGFormer) |
| **MotionBERT** — second backbone, preprocessed H36M | Zhu, Ma, Liu, Liu, Wu & Wang, *MotionBERT: A Unified Perspective on Learning Human Motion Representations*, ICCV 2023. [arXiv](https://arxiv.org/abs/2210.06551) · [code](https://github.com/Walter0807/MotionBERT) |
| **VideoPose3D** — 2D-to-3D lifting paradigm | Pavllo, Feichtenhofer, Grangier & Auli, *3D Human Pose Estimation in Video with Temporal Convolutions and Semi-Supervised Training*, CVPR 2019. [arXiv](https://arxiv.org/abs/1811.11742) |
| **P-STMO** — MPI-INF-3DHP preprocessing | Shan, Liu, Zhang, Wang, Wang & Ding, *P-STMO: Pre-Trained Spatial Temporal Many-to-One Model for 3D Human Pose Estimation*, ECCV 2022. [code](https://github.com/paTRICK-swk/P-STMO) |
| **3DPCNet** — closest prior work, learned canonicalization | Ekanayake et al., *3DPCNet: Estimator-Agnostic Canonicalization of 3D Human Pose*, ICASSP 2026. [arXiv](https://arxiv.org/abs/2509.23455) |
| **TRIAD** — the frame construction | Black, *A Passive System for Determining the Attitude of a Satellite*, AIAA Journal 2(7), 1964 |
| **TRIAD covariance, primary-axis rule** | Shuster & Oh, *Three-Axis Attitude Determination from Vector Observations*, J. Guidance and Control 4(1), 1981 |
| **Wahba's problem** | Wahba, *A Least Squares Estimate of Satellite Attitude*, SIAM Review 7(3), 1965 |
| **Landmark error → frame orientation** | Della Croce, Cappozzo & Kerrigan, *Med. Biol. Eng. Comput.* 37(2), 1999; Della Croce, Leardini, Chiari & Cappozzo, *Gait & Posture* 21(2), 2005 |
| **YOLOv8-pose** — 2D front end | Jocher, Chaurasia & Qiu, *Ultralytics YOLOv8*, 2023 |

---

## Citation

```bibtex
@mastersthesis{khan2026viewinvariant,
  title  = {Reliability-Aware View-Invariant 3D Human Pose Estimation
            from a Frozen Monocular Estimator},
  author = {Mehedi Hasan Khan},
  school = {Comilla University},
  year   = {2026}
}
```

Please also cite the backbone you use — MotionAGFormer (Mehraban et al., WACV 2024) or MotionBERT (Zhu et al., ICCV 2023).

## License

Academic research use. The underlying MotionAGFormer code follows the license of the original repository.

## Acknowledgement

Built on the official [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer) implementation, with data preprocessing from [MotionBERT](https://github.com/Walter0807/MotionBERT) and [P-STMO](https://github.com/paTRICK-swk/P-STMO), and 2D detection from [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics). Thanks to the authors for releasing their code.
