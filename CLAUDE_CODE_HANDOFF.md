# Handoff: Mehedi's BSc thesis — Comilla University, CSE

Handoff for Claude Code. Written 2026-08-07, the night before the final
pre-defence sprint. Everything below is verified against the repo state at
commit `b4d86e8` (working tree clean).

---

## 1. The project in one paragraph

A training-free, reliability-aware **geometric canonicalization** framework for
cross-view comparability of frozen monocular 3D pose predictions. The core is a
**body-fixed anatomical reference frame** built from predicted joints — this is
the **TRIAD algorithm (Black 1964)** with the torso axis primary, plus an
**analytic reliability score** (geometric mean of six plausibility components)
and **calibration-free multi-view fusion**. Everything added on top of the
frozen estimator (MotionAGFormer-XS, MotionBERT) has **zero trained
parameters**. Two datasets: MPI-INF-3DHP (development), Human3.6M (held out,
180 camera pairs). **Defense: 9 Aug 2026.**

The thesis's honest position: a **Kabsch-to-template baseline beats the
anatomical frame 180/180** on the headline metric, and this is reported rather
than hidden. The contribution is the *experimental boundary* of how far the
geometry carries + a pre-registered **failure-support map and routing rule**
that combine the two alignments so the *combination* is never worse than the
better single one.

## 2. The evidence discipline (this is what the examiner tests)

- **Every experiment is pre-registered**: criterion + all readings committed to
  git **before** the run. Commit history shows pre-registration commits
  (`5dbc47a`, `4ccee2f`) preceding result commits (`2a97b2e`, `027dc13`).
- **258/258** numerical claims recomputed from stored artifacts by
  `python -m evaluation.audit_numbers`; **76+ unit tests** pass.
- **No number in the report was typed by hand** — all trace to
  `thesis_artifacts/*.json` / `.npz`.

## 3. The novelty (3 post-freeze pre-registrations, all Reading 1)

### Exp 12 — Anchor corruption (`evaluation/anchor_corruption.py`, `thesis_artifacts/anchor_corruption/`)
Corrupt the frame's own support joints {hips, thorax} in the lifted 3D pose:

| σ (mm) | XS anatomical | XS template17 | MB anatomical | MB template17 |
|---|---|---|---|---|
| 0 | 53.45 | 43.30 | 44.13 | 40.96 |
| 80 | 215.66 | 49.15 | 217.81 | 47.15 |
| 160 | 337.87 | 63.22 | 339.86 | 61.63 |

The frame **collapses 53→338 mm**; the 17-joint Kabsch fit degrades
**43→63 mm**. Combined with the (failed) distal-corruption experiment, the two
alignments have **disjoint failure supports**.

### Exp 13 — Routing rule (`evaluation/selection_rule.py`, `thesis_artifacts/selection/`)
Fixed rule: *use the anatomical frame iff core confidence ≥ 0.7 AND distal
confidence < 0.7, else Kabsch*. Never worse than the better single alignment at
**19/20** (regime, σ, backbone) cells; trails **5.4 mm** in one transition cell
(tolerance 7 mm, pre-registered); beats Kabsch alone by **38.8 mm (XS) /
47.3 mm (MB)** at σ=160 distal. Uses Kabsch at clean data.

### Exp 14 — 2D-input invariance (`evaluation/misdetect_invariance.py`, `thesis_artifacts/misdetect/`)
Through the **real** detection path (YOLOv8 → frozen lifter; clean re-lift
anchors to the cache at **0.00 mm**):
- 2D displacement of distal/core joints up to **434 px (21% of frame, 0.6 in
  normalized input space)** moves the lifted pose **≤ 0.33 mm mean** (per-joint
  worst 0.77 mm). The lifter absorbs 2D keypoint error entirely.
- Real detector confidence is **flat**: mean 0.74/0.81, **every frame < 0.9** —
  no threshold can separate frames on clean data. The measured signal that
  does vary is the analytic reliability score (cam0 min 0.0 vs cam1 min 0.854).
- **Verdict: Reading 1** — the failure surface is at the 3D alignment level,
  end to end; the routing rule's simulated confidence gate is explicitly a
  stand-in for a channel with no usable signal.

**Known caveat (documented, not hidden):** the routing rule's confidence signal
is **simulated** (a calibrated function of injected noise), not measured — an
upper bound on a real channel. Experiment 14 measured the real channel and
showed it is flat on clean data.

## 4. Supervisor's concern, answered

"Where is the comparison between the base output and your proposal?" — it is
report §5.2–5.3 + Appendix A of the minimal report, and collected in
`thesis_report/SIR_CONCERNS.md` (EN + BN):

| | Base | Proposed |
|---|---|---|
| MPJPE (H36M) | 45.149 mm | 45.149 mm (identical by construction) |
| Cross-view distance (13 joints) | 372.7 mm | **93.4 mm (−72.2%)** |
| Pairs improved | — | 179/180 |
| Kabsch-to-template baseline | — | **57.5 mm, wins 180/180 (honest third row)** |

## 5. Files created/changed this session

**In `MotionAGFormer/` (own git repo, 14 pre-registrations):**
- `evaluation/anchor_corruption.py`, `evaluation/selection_rule.py`,
  `evaluation/misdetect_invariance.py` — the three new experiments
- `thesis_artifacts/anchor_corruption/`, `thesis_artifacts/selection/`,
  `thesis_artifacts/misdetect/` — PREREGISTRATION.md + RESULT.md + JSON each
- `thesis_report/Minimal_Thesis_Report.tex` (+`.pdf`) — **the submission doc,
  28 pages, compiles clean** (0 errors/overfull/underfull/undefined)
- `thesis_report/SIR_CONCERNS.md` — supervisor concerns + comparison answers
- `thesis_report/NOVELTY_PLAN.md` — the 2-night plan, claims + anti-claims
- `REPO_MAP.md` — updated with experiments 12–14

**At repo root `/e/thesis/` (NOT a git repo — see §7):**
- `README.md` (standard GitHub), `ROADMAP.md` (file-by-file),
  `.gitignore`
- `_archive/early_work/` — superseded MotionBERT/MLP work moved (not deleted)
- `_archive/tooling/` — node/agent tooling junk moved

## 6. How to run everything

```bash
cd /e/thesis/MotionAGFormer
./venv/Scripts/python.exe -m evaluation.audit_numbers          # 258/258
./venv/Scripts/python.exe -m unittest discover -s tests -q     # OK
./venv/Scripts/python.exe -m evaluation.anchor_corruption --selfcheck
./venv/Scripts/python.exe -m evaluation.selection_rule --selfcheck
./venv/Scripts/python.exe -m evaluation.misdetect_invariance --selfcheck
# report:
cd thesis_report && pdflatex Minimal_Thesis_Report.tex   # run TWICE
```

Windows paths: venv is `MotionAGFormer/venv/Scripts/python.exe`; MPI data at
`E:/Thesis/mpi_inf_3dhp`; bash syntax only (no cmd/PowerShell).

## 7. Things Claude Code must know before touching anything

1. **The root `/e/thesis` is not a git repo** (its `.git/` is broken tooling
   state). The real history lives in `MotionAGFormer/`. Before any GitHub push:
   either `git init` at root (recommended: init at root and add MotionAGFormer
   as the main tree, or push `MotionAGFormer/` alone — it has the full
   pre-registration history).
2. **Pre-registration order is sacred.** Any new experiment: write
   PREREGISTRATION.md + commit, THEN run, THEN commit the result. Never the
   reverse — the commit ordering is defense evidence.
3. **Do NOT** retrain, tune the routing threshold, add more experiments, or
   touch the frozen report's 258 claims. The map is complete; every extra run
   is another number to defend.
4. **Honest framing is the thesis.** Never claim the frame wins — claim the
   *combination* is never worse. Keep every "honest boundary" paragraph.
5. **P2 of exp 14 was amended** (documented in its PREREGISTRATION.md): the
   cached `components[:,0]` is a reliability component, not detector
   confidence. Don't reintroduce that mistake.
6. `Minimal_Thesis_Report.pdf` is committed; the `.tex` is the source of truth
   (compile with pdfLaTeX ×2).

## 8. Remaining work (Night 2, 8 Aug)

- [ ] Read `Minimal_Thesis_Report.pdf` aloud once (28 pp).
- [ ] Rehearse `SIR_CONCERNS.md` answers + the 3DPCNet answer + the
      "cross-view agreement is my protocol, not a named standard" answer.
- [ ] Re-run audit + tests once more after any edits.
- [ ] Send the supervisor email (`thesis_report/SUPERVISOR_EMAIL.md`).
- [ ] Push to GitHub (resolve §7.1 first).
