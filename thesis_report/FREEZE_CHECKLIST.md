# Freeze checklist

Run top to bottom before submitting. Every item has a command or a specific
thing to look at; nothing here is "check it looks right".

## 1. Numbers

- [ ] `python -m evaluation.audit_numbers` → **167/167**, exit 0.
      Fails loudly if any reported figure drifts from its JSON artifact.
- [ ] `python -m unittest discover -s tests -q` → **72 tests, OK**.
- [ ] Spot-check by hand that the two averaging conventions still disagree the way
      §5.10 says: per-pair 74.1 % against ratio-of-means 76.5 %. If someone
      "fixes" one of them the disclosure becomes false.

## 2. Figures

- [ ] `python -m presentation.render --twoview --teaser` regenerates all seven
      generated figures without error.
- [ ] **Open each PNG and look at it.** Past failures this catches: skeletons
      rendered upside down (raw poses are y-down), the hip axis clipped out of
      frame by equal aspect, a null result made to look dramatic by a 2 mm y-axis,
      and panels at different scales making the canonical column look tighter than
      it is.
- [ ] `fig_teaser.png` reads in under ten seconds with no caption.
- [ ] Every figure's numbers match the artifact it was drawn from — they are read
      from the same JSON, so this is a check that the JSON is current.

## 3. The report compiles clean

- [ ] `pdflatex` **twice**, then grep the log for `undefined`, `LaTeX Error`,
      `Warning: Reference`. All must be empty.
- [ ] **Abstract ends on page iii.** It has spilled onto page iv five times during
      editing. Check `p4` starts with `ACKNOWLEDGEMENT`.
- [ ] Page count recorded: **87**.
- [ ] Table of contents regenerated after the last edit.

## 4. Citations

- [ ] All 24 references cited at least once; no orphans.
      `[n]` appearing in text for every n in 1..24.
- [ ] README citations match the thesis bibliography exactly. This was wrong
      before: the README had VideoPose3D under the wrong title and author,
      MotionBERT under the wrong year, and P-STMO under a wrong name.
- [ ] Refs [19]–[24] (TRIAD, Shuster & Oh, Wahba, Markley, Della Croce ×2) are the
      prior-art block; confirm §2.5 attributes each correctly.

## 5. Claim consistency

Read these side by side and confirm they say the same thing about scope:

- [ ] Abstract → §1.3 Objectives → §1.4 Contributions
- [ ] §2.5 (what is inherited) → §3.2 (the relation) → §6.1 Discussion
- [ ] Table 5.1 verdict column → the section each row points to
- [ ] Nothing anywhere still says the propagation is **derived** by us, or calls
      the axis-length relation the most transferable finding. Grep: `we derive`,
      `derived from first principles`, `most transferable`.

## 6. Honesty markers still present

These are the sentences that make the thesis defensible. Confirm each survived
the last round of edits:

- [ ] The withdrawal of the "constants agree to one percent" argument (§5.16).
- [ ] The averaging-convention disclosure (§5.10).
- [ ] The reliability-triage result labelled **exploratory** everywhere it appears.
- [ ] The thorax/hip pinning artifact disclosed (§5.19) — our own figures overstate
      per-joint performance without it.
- [ ] The retrieval result marked as superseded protocol, not a measurement.
- [ ] §5.16's note that criterion (c) was tightened post hoc, and why it cannot
      flip the verdict.

## 7. Pre-registrations

- [ ] Four `PREREGISTRATION.md` files under `thesis_artifacts/`: tta,
      multilandmark, conditioning, radial.
- [ ] `git log --follow` on each shows the pre-registration commit **precedes**
      the commit containing its result. This is the whole point; verify it.

## 8. Artifacts

- [ ] Every JSON under `thesis_artifacts/` referenced by `audit_numbers.py` exists
      and is committed.
- [ ] No absolute paths from this machine baked into any artifact or script.
- [ ] `preds.npz` and `preds_motionbert.npz` are present or their absence is
      documented (they are large; the audit reads the derived JSONs, not the
      caches).

## 9. Appendix

- [ ] Appendix A tables regenerate from the artifacts and match Chapter 5.
- [ ] Per-action and per-pair tables have the same totals as the summary rows.

## 10. Repository

- [ ] README renders on GitHub; the teaser image path resolves.
- [ ] `python -m evaluation.audit_numbers` is the first command a visitor sees.
- [ ] No stray logs, checkpoints or `.npz` files newly tracked:
      `git ls-files | grep -E '\.(pt|pth|log|bin)$'`
- [ ] Working tree clean; everything committed.

## 11. Submission

- [ ] Front matter correct: title, name, ID 12108004, session 2020–21,
      supervisor, chairman of the examination committee (Dr. Md. Faisal Bin Abdul
      Aziz, Associate Professor), university logo present.
- [ ] Supervisor emailed about the TRIAD and biomechanics positioning **before**
      he receives the PDF (`SUPERVISOR_EMAIL.md`).
- [ ] University AI-use policy checked and any required disclosure included.
- [ ] `DEFENSE_QA.md` read once end to end.
