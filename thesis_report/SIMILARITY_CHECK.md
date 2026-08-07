# Similarity and AI-disclosure check

Run 6 August 2026 against `Full_Thesis_Report.tex` at commit `1073b5c` (56 pages,
28 references). This is a pre-submission self-check, **not** a Turnitin report.
Read the limits section before quoting any of it to anyone.

---

## 1. What was checked, and what came back

### Direct quotation audit — clean

Every `` `` … '' `` pair in the source was extracted and classified. There are 32.
Twenty-eight are paper titles inside the reference list, which is where they
belong. The remaining four are in the body:

| Quoted string | What it is |
|---|---|
| "A Lightweight, Training-Free, Reliability-Aware Geometric Canonicalization Framework…" | the thesis's own title *as it stood at this scan; "Reliability-Aware" was removed from the title on 8 Aug 2026* |
| "MEHEDI HASAN KHAN" | the author's own name |
| "Hips (the current frame)" | a row label in Table 5.7 |
| "Hips alone" | a row label in Table 5.7 |

**The body contains no quoted material from any source.** Every description of
prior work is paraphrased in the author's own words and carries a numbered
citation. This removes the single most common cause of a genuine plagiarism
finding: a sentence lifted from an abstract and left without quotation marks.

### Verbatim web search — no matches

Eight distinctive sentences were searched as exact phrases, weighted toward
Chapter 2 and the abstract, which is where close paraphrase of a cited paper is
most likely:

1. "infinitely many 3D configurations project to the same 2D observation"
2. "Accuracy and comparability are separate properties"
3. "a network of two residual blocks" + "ground-truth 2D keypoints"
4. "Knowing that data is out of distribution… which predictions to distrust"
5. "Learned uncertainty attaches a variance head or an ensemble to the estimator"
6. "whose literature establishes that the better-known direction should be primary"
7. "not which frame to trust within a construction" + "articulation dominates"
8. the full title as a phrase

**No sentence returned a verbatim match.** Every result was topically related
work — the concepts are standard, which is expected in a literature review; the
wording is not anyone else's.

### Attribution audit — unusually strong

This is the part that would matter in a viva, and the report is in better shape
here than most:

- §2.6 names 3DPCNet, VI-HC (Wei et al.), MoViD, V-VIPE, CMANet, conformal
  keypoint detection, BLAPose and DDHPose, each with its gap stated.
- The frame construction is identified as **TRIAD** [19] rather than presented
  as the author's, and the primary-axis rule is credited to **Shuster and Oh**
  [20].
- The landmark-error propagation is credited to **Della Croce, Cappozzo and
  Kerrigan** [23] and **Della Croce et al.** [24], with an explicit sentence
  saying an earlier draft had wrongly claimed it.
- **Wei et al.** [25] is flagged as the closest precedent, with the note that a
  reader who found it independently would reasonably ask why it was absent.

A report that names its own nearest competitor and retracts its own earlier
novelty claim is not the profile of a document with an attribution problem.

---

## 2. Limits — read this before relying on any of the above

This check **does not replicate Turnitin** and cannot. Turnitin matches against
a subscription-journal index and, more importantly, an archive of previously
submitted student papers. **Neither is reachable from here.** A clean result
above therefore means "no match on the open web for the phrases tested", not
"will score low". Eight sentences out of a 56-page document is a spot check, not
a sweep.

If the department offers a draft/self-check Turnitin submission, use it. That is
the only way to get the real number before it counts.

## 3. Matches you should expect, and should not worry about

Flag these to your supervisor in advance. A similarity score that has been
explained beforehand is a formality; the same score raised for the first time in
a viva is an argument.

| Source of match | Why it is legitimate |
|---|---|
| Declaration, certificate and acknowledgement pages | Departmental template wording, shared by every thesis in the department |
| The 28-entry reference list | Titles and venues must match the originals exactly |
| Dataset and metric names | "MPI-INF-3DHP", "Human3.6M", "mean per-joint position error", "Procrustes", "Gram-Schmidt", "Kabsch" have no synonyms |
| Standard equations | The TRIAD construction, the Kabsch/Procrustes formulation and the Spearman coefficient are written the only way they are written |
| Your own prior submissions | If a proposal or progress report was submitted through Turnitin, it is in the archive and **will** match this report heavily. Say so in advance — this is the most common cause of an alarming score on a final thesis |

## 4. AI detection — the position taken here, stated plainly

**No prose in this report has been rewritten to lower an AI-detector score, and
none will be.** Two reasons, both practical rather than moral:

1. The text describes what was actually done, by whom, including five failed
   pre-registered experiments, a sixth that returned a competing method as
   better, and a retraction. Editing it to alter an
   authorship signal would make it less accurate in exchange for a number that
   no examiner is entitled to treat as evidence on its own.
2. AI detectors have a well-documented false-positive problem on exactly this
   register — dense, hedged, technical writing with consistent sentence
   structure. Dodging one costs the precision that makes the report defensible,
   and the precision is the thing being examined.

**The defence, if the question is raised, is the repository, and it is a strong
one.** This project can demonstrate its own provenance in a way that very few
undergraduate theses can:

- Seventeen pre-registration files, each committed **before** its experiment ran, in
  a git history with timestamps.
- `evaluation/audit_numbers.py` re-derives all 304 numerical claims from stored
  JSON artifacts; it passes.
- 76 unit tests.
- Every figure generated by a script from those same artifacts, including the
  two corrected in commit `1073b5c`.
- More than half of the seventeen pre-registered experiments failed and are
  reported as failures, and one returned a competing baseline that beats the
  proposed method on every one of 180 pairs and all 15 actions.

No detector output outweighs a timestamped record of the work. Offer to walk
through the git log.

## 5. Action items before submission

- [ ] **Read Comilla University CSE's rule on AI assistance.** This is yours to
      check, not mine to assume. If disclosure is required, §6 below is a draft.
- [ ] Submit to the department's draft Turnitin box if one exists.
- [ ] Tell your supervisor in advance about the template pages, the reference
      list, and any earlier proposal already in the Turnitin archive.
- [ ] Have the git log ready to open on the defence machine.

## 6. Draft disclosure paragraph — use, edit, or discard

Only include this if the university's policy requires disclosure. It is written
to be accurate; do not soften it, and do not include it if it is not required.

> **Declaration on the use of AI tools.** An AI coding assistant was used during
> this project for software development, for figure generation, and for editorial
> work on this report. All experimental design, pre-registration criteria and
> interpretation of results are the author's own. Every numerical claim in this
> report is re-derived from stored artifacts by an automated audit
> (`evaluation/audit_numbers.py`, 304 claims) and is reproducible from the
> repository, and the pre-registration files were committed to version control
> before the corresponding experiments were run. No result, measurement or
> citation in this report was generated without being verified against its
> source.

---

*Regenerate the quotation audit and sentence list with the commands recorded in
this session; the searches must be re-run by hand.*
