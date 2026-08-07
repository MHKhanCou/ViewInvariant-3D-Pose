OVERLEAF UPLOAD — Full_Thesis_Report
=====================================
Preamble, title page, bonafide certificate, chapter styling and reference
format are matched to the departmental example report.

CONTENTS
  Full_Thesis_Report.tex   the report
  appendix_tables.tex      Appendix A, GENERATED - do not hand-edit
  images/                  6 result figures (referenced by the .tex)
  README.txt               this file

appendix_tables.tex is produced by
  python -m evaluation.make_appendix_tables
from the JSON artifacts. Editing it by hand would break the guarantee that
the appendix matches the audited numbers. Re-run the script instead.

HOW TO COMPILE
  1. overleaf.com -> New Project -> Upload Project -> upload the zip
  2. Compiler: pdfLaTeX  (Menu -> Compiler)
  3. Compile TWICE so the table of contents and cross-references resolve.

--------------------------------------------------------------------
YOU MUST DO THESE THREE THINGS
--------------------------------------------------------------------

1. ADD THE UNIVERSITY LOGO
   The title page expects:   images/university logo.jpg
   (exact filename, with the space — same as the example report).
   Copy it from the example project. Without it the build FAILS.

2. CONFIRM THE SESSION YEAR
   Title page currently says:  Session: 2020-21
   This was inferred from your ID (12108004) against the example
   (ID 12008040 = Session 2019-20). CHECK IT and correct if wrong.

3. COMPLETE TWO REFERENCES
   [6] CMANet  and  [7] BLAPose  have no author lists, and
   [9] Khanal and Zhou has no venue. These were left incomplete on
   purpose rather than invented. Fill them from the actual papers.

--------------------------------------------------------------------
ALREADY FILLED IN
--------------------------------------------------------------------
  Name        MEHEDI HASAN KHAN
  ID          12108004
  Dept        Computer Science and Engineering
  University  Comilla University :: Comilla-3506
  Supervisor  Dr. Mahmudul Hasan, Associate Professor
              (also listed as Chairman, Exam Committee, as in the example)
  Date        August 2026

--------------------------------------------------------------------
NOTES ON THE CONTENT
--------------------------------------------------------------------
- Every number in Chapter 5 comes from a JSON artifact under
  MotionAGFormer/thesis_artifacts/ and is verified by
  `python -m evaluation.audit_numbers`  (304 claims, all passing).
  The test suite is 76 tests, all passing.

- THE HEADLINE IS NOW THE CROSS-DATASET REPLICATION (Section 5.9).
  Canonicalization was re-tested on Human3.6M, which played no part in
  developing the method, so all 180 camera pairs are held out. It reduces
  cross-view distance by 74.1% (against 32.4% on MPI-INF-3DHP), improves
  179 of 180 pairs, and closes 90.5% of the gap to a Procrustes oracle.
  This is the strongest single result in the report. Lead with it.

- The report deliberately includes negative results: the withdrawn
  view-selection claim, the five falsifications of the reliability score,
  the Human3.6M retraction of the bone-length signal (5.8), and the
  retrieval regression. Chapter 6's argument depends on them.

- Section 5.8 RETRACTS the report's own bone-length result. On
  MPI-INF-3DHP that signal scored rho = +0.492; on Human3.6M it scores
  +0.098 and fails all five criteria. Do not quote +0.492 on its own.
  The section is defensible, not apologetic: the evaluation pipeline is
  verified against the backbone's own eval script to three decimal places
  (45.149 mm), and the obvious excuse -- "Human3.6M is in-domain so there
  is no error to predict" -- is tested and rejected.

- Section 5.10 refines rather than retracts the fusion claim: at four
  views an unweighted mean is WORSE than one arbitrary view, while a
  median is better. 67.9% of frames improve either way; the mean is
  dragged down by a catastrophic minority. Note that the plain median
  uses no reliability score and beats the reliability-weighted mean.

- Section 5.11 (multi-scale on Human3.6M) contains a finding worth
  leading with in the viva. Reporting the five levels separately showed
  the two arms agreeing (57.1 vs 57.6 mm) while the two legs did not
  (69.4 vs 30.7 mm). The cause was in our own code: the legs were
  defined with different axes, the left using the short root-to-hip
  vector as its primary axis. Defining both alike drops the left leg to
  29.3 mm, matching the right, and raises the multi-scale improvement
  from +25.6% to +37.2% with all 180 pairs improving.

  Why this is a strength, not an admission: the sensitivity argument in
  Section 3.2 PREDICTS a shorter primary axis gives a noisier frame, so
  the defect was explained before it was found. Four audit claims assert
  the fix is surgical -- torso and both arms are bit-identical. And it
  adds no parameters and no labels; it only makes two definitions agree
  with the four that were already consistent.

--------------------------------------------------------------------
THE "IT USES PRETRAINED MODELS" OBJECTION
--------------------------------------------------------------------
This is the most likely question in the viva. Section 3.1 answers it
with a per-component table, and Section 3.1 now also states it in
operational terms. Memorise this answer:

  The pipeline is NOT training-free and the report says so on its first
  methodology page. The detector and the lifting network are trained by
  their authors and used frozen. What is training-free is everything
  this project ADDS: zero learned parameters.

  The property claimed is a property of the DEPLOYMENT REQUIREMENT, not
  of the pipeline's history. To apply this framework to a new estimator,
  a new camera or a new domain you need no training data, no labels, no
  calibration and no gradient step. To apply 3DPCNet -- the closest
  competitor, also estimator-agnostic -- you must train its network.
  Both sit downstream of a trained estimator. The difference is entirely
  in what the USER must supply.

  What would falsify the claim is not the presence of a pretrained
  backbone, which is assumed. It is the method needing per-dataset
  tuning. The Human3.6M results are the evidence: constants were fixed
  on MPI-INF-3DHP, never adjusted, and applied to 180 unseen pairs.

Do NOT say "no training" as shorthand. Say "adds zero trained
parameters". The shorthand is what invites the objection.

- Section 3.1 states plainly that the pipeline is NOT training-free: the
  backbone and detector are trained by their authors. Keep it. It is the
  first thing a careful examiner will ask about.

- Section 2.3 states that geometric canonicalization is NOT claimed as
  novel, because 3DPCNet (2025) addresses the same problem and reports a
  similar hand-built baseline performing worse. This is deliberate; do
  not upgrade it back into a novelty claim.
