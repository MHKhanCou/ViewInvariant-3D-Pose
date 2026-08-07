# Repository roadmap — what every file is for

Read this before the viva. If an examiner asks *"where is X?"*, the answer is on
this page. Ordered by how likely you are to need it, not alphabetically.

---

## The five files that matter most

| # | File | Why it matters |
|---|---|---|
| 1 | `thesis_report/Full_Thesis_Report.tex` | The report itself, 1690 lines. Compile with `pdflatex` twice — **not** `latexmk`, no Perl on this machine |
| 2 | `canonical/body_frame.py` | The entire method. The thesis is this file plus what was learned from testing it |
| 3 | `evaluation/h36m_crossview.py` | The central result: 372.7 → 93.4 mm |
| 4 | `evaluation/template_baseline.py` | The baseline that beats you. Know this one cold |
| 5 | `evaluation/audit_numbers.py` | Re-derives all 273 reported claims from stored artifacts |

**One command proves the whole report is internally consistent:**

```bash
./venv/Scripts/python.exe -m evaluation.audit_numbers        # 273/273, exit 0
./venv/Scripts/python.exe -m unittest discover -s tests -q   # 76 tests
```

---

## The method

```
canonical/
  body_frame.py        Gram-Schmidt body frame (TRIAD). y = P[8]-P[0], x_raw = P[1]-P[4].
                       Reads FOUR joints: {0 root, 1 r_hip, 4 l_hip, 8 thorax}.
  multiscale.py        per-limb frames. Circular by construction - demoted to exploratory.
  multilandmark.py     alternative frame constructions (the shoulder-axis variant that wins).
```

Everything else in the repo either feeds this file or measures it.

---

## The experiments — one module, one JSON artifact, one report section

Every module writes to `thesis_artifacts/<name>/` and every number in the report
is read back from there by the audit. **No number in the report was typed by
hand.**

### The results that hold

| Module | Artifact | What it shows |
|---|---|---|
| `h36m_crossview.py` | `h36m_crossview/` | The central claim, all 17 joints |
| `h36m_noncon.py` | `noncon/` | The same, scored off the 4 joints the frame builds from — **the headline, 72.2 %** |
| `h36m_motionbert.py` | `h36m_motionbert/` | Second backbone, 75.8 % |
| `multilandmark_eval.py` | `multilandmark/` | Level 1: longer axis wins **(the one confirmation)** |
| `conditioning_abstention.py` | `conditioning/` | The falsified score gates canonicalization quality (exploratory) |

### The results that failed — half the thesis

| Module | Artifact | How it failed |
|---|---|---|
| `template_baseline.py` | `template/` | **A simpler method beats ours, 180/180 pairs** |
| `template_ablation.py` | `template_ablation/` | …under three unrelated templates too |
| `translation_ablation.py` | `translation_ablation/` | …under every centring |
| `template_action.py` | `template_action/` | …on all fifteen actions |
| `radial_law.py` | `radial/` | Level 3: radius does not predict per-joint disagreement |
| `h36m_replication.py` | `h36m_replication/` | Bone-length signal retracted, ρ +0.492 → +0.098 |
| `tta_consistency.py` | `tta/` | Failed all three criteria |
| `multiscale_control.py` | `h36m_multiscale/` | Circularity control that demoted the 55.1 % figure |
| `occlusion_robustness.py` | `occlusion/` | The tenth, run after the freeze. Failed |
| `template_mismatch.py` | `mismatch/` | The eleventh. Failed - the baseline needs no matching template |
| `anchor_corruption.py` | `anchor_corruption/` | The twelfth (post-freeze). **Holds** - corrupt the frame's support joints {1,4,8} and it collapses 53.45 -> 337.87 mm while the 17-joint fit degrades gracefully |
| `selection_rule.py` | `selection/` | The thirteenth (post-freeze). **Holds** - confidence-gated routing is never worse than the better single alignment, 38-47 mm better than template alone at sigma=160 distal |
| `misdetect_invariance.py` | `misdetect/` | The fourteenth (post-freeze). **Holds** - the frozen lifter is invariant to 2D keypoint displacement through the real detection path (<=0.33 mm at 434 px), and the real confidence channel is flat (all frames < 0.9) |

### Supporting

`fusion.py`, `fusion_eval.py`, `h36m_fusion.py` (multi-view fusion) ·
`reliability.py` (the six-component score) · `oracle.py` (Procrustes floor) ·
`cost_benchmark.py` (402 FLOPs) · `metrics.py`, `protocol.py` (shared plumbing) ·
`make_*_figures.py`, `make_appendix_tables.py` (everything printed, from artifacts)

---

## The written material

```
thesis_report/
  Minimal_Thesis_Report.tex the ~20-page submission report
  Full_Thesis_Report.tex     the extended report
  SIR_CONCERNS.md            supervisor feedback, answered (base vs proposal included)
  NOVELTY_PLAN.md            the two-night novelty plan and exact claims
  appendix_tables.tex        GENERATED - do not hand-edit
  WORKFLOW.md                RGB video -> comparison, end to end. Sir's two questions
  SIR_QA.md                  Sir's questions and the likely follow-ups, EN + BN
  SUPERVISOR_EMAIL.md        the email to send before he reads the report
  DEFENSE_QA.md              the long viva prep: 10s/30s/2min/5min pitches, hard questions
  EXPLAIN.md                 plain-language, EN + BN
  EXPLAIN_SIMPLE.md          jargon-free, EN + BN
  FREEZE_CHECKLIST.md        pre-submission gate
  images/                    25 figures, all generated by presentation/render.py
```

```
thesis_artifacts/
  <experiment>/PREREGISTRATION.md   eleven of them, each committed BEFORE its run
  occlusion/RESULT.md               the tenth, and why it failed
  defense_deck.html                 16 slides
  planning/                         superseded working notes - do not quote these
```

**`planning/` is stale by design.** It contains earlier drafts, including numbers
the examiner invalidated. If a document in there disagrees with the report, the
report wins.

---

## The demo

```
app.py               Gradio interface. Coordinate System toggle = the thesis, live
demo_live/lifter.py  the shared lifting path - the SAME code the evaluation uses
presentation/render.py   every report figure, from artifacts
```

The evaluation and the demo share `demo_live/lifter.py` on purpose, so the
demonstrated behaviour and the reported numbers cannot diverge.

---

## Where the big files live, and why they are not on GitHub

| Directory | Size | Contents |
|---|---|---|
| `thesis_artifacts/h36m_replication/` | 228 MB | Cached H36M predictions (`preds.npz`) |
| `thesis_artifacts/h36m_motionbert/` | 128 MB | Same, second backbone |
| `thesis_artifacts/benchmark/` | 108 MB | Benchmark run outputs |
| `thesis_artifacts/cross_view_eval/` | 2.6 MB | MPI cache — **the template baseline reads this** |

These are gitignored. Every experiment reads its predictions from them, which is
why re-running an analysis takes seconds rather than hours. **Do not delete
them before the defence.**

---

## If an examiner asks…

| Question | Open this |
|---|---|
| "Show me the method" | `canonical/body_frame.py`, 40 lines |
| "How do I know the numbers are real?" | Run `audit_numbers.py`, then `git log` for the pre-registration timestamps |
| "Where is the base vs proposed comparison?" | `thesis_artifacts/figures/fig_realview.png` and Table `tab:h36mcv` |
| "You said a baseline beats you — where?" | Report the section "A Single-View Baseline, and It Wins", `thesis_artifacts/template/template.json` |
| "What did you pre-register?" | `ls thesis_artifacts/*/PREREGISTRATION.md` — eleven files, all timestamped before their results |
