# Pre-registration: cross-view distance over the joints the frame is not built from

Written and committed **before** the recomputation is run, so version history
shows the specification preceding the number. Same practice as
`thesis_artifacts/tta/`, `multilandmark/`, `conditioning/` and `radial/`.

Date: 2026-08-06

## Why

Section 5.16.2 establishes that the Gram-Schmidt frame pins the joints it is
built from. The thorax canonicalizes at 22.1 mm and the hips at 54.4 mm against
197.5 mm for articulated joints, not because the method succeeds on them but
because the construction fixes their position by definition. That section then
says the report separates them "explicitly", which was true of the per-joint
table and **false of the headline**: the 74.1 percent improvement and the 75.3 mm
canonical distance are still averages over all seventeen joints.

An external reviewer identified this as the most exposed number in the report. We
agree. This recomputes it over the joints that are not part of the construction.

## The joint sets, stated by index rather than by name

The frame is built from the vertical vector `P[8] - P[0]` and the lateral vector
`P[1] - P[4]`, so the **constructor set is exactly `{0, 1, 4, 8}`**: root,
right hip, left hip, thorax. The **retained set is the remaining thirteen
indices**, `{2, 3, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16}`.

"Non-constructor joints" is not used as a specification anywhere in the code or
the report without this list beside it, because the phrase is meaningless to a
reader who has not memorised equation 3.1.

## What joint 0 does and does not affect

Joint 0 is identically zero in root-relative coordinates. The raw and canonical
cross-view distances are means over per-joint Euclidean distances, so joint 0
contributes a zero term to each; dropping it rescales both by 17/16 and leaves
**their ratio, the improvement percentage, unchanged**.

This does not extend to the oracle. `evaluation/oracle.py:18` centres on the
centroid of the point set, computing `source.mean(axis=0)` before forming the
covariance, so removing a point at the origin shifts the centroid, changes the
covariance and changes the fitted rotation. **The oracle distance and the
oracle-gap-closed percentage will move.**

Consequently the change in the improvement percentage is driven by joints 1, 4
and 8 alone — three joints — while the change in the oracle gap is driven by all
four. Both are reported.

## No expected direction

We do not predict whether the improvement will rise or fall, and no criterion
here treats either outcome as a success.

Excluding these joints raises the canonical distance, since the pinned joints
were dragging it down. It also raises the raw distance, since the hips and
thorax are among the joints two cameras disagree about least before
canonicalization. The improvement is a ratio of the two, and which term moves
further depends on how much of the raw disagreement at those joints is camera
rotation as opposed to articulation. **We have not measured that**, so we have no
basis for a prediction and record none. Whatever the number is, it is the number,
and it is the number the report will lead with.

Nothing in the verification for this experiment may check whether the figure
fell. An early draft of the plan for tonight instructed exactly that, which would
have made a disconfirming result look like a bug in the joint mask — the failure
mode Section 5.14.1 exists to catch.

## What is checked instead

- The excluded indices are exactly `{0, 1, 4, 8}` and thirteen are retained.
- A unit test asserts that including or excluding joint 0 leaves the raw and
  canonical distances' ratio identical, **scoped to those two** and not to the
  oracle, for the reason given above.
- Both backbones are run, with the same cluster bootstrap over the thirty
  subject-action groups used everywhere else.

## Reporting

The non-constructor figure becomes the headline in the abstract, Section 1.4,
Table 5.1 and Section 5.10, with the seventeen-joint figure given alongside and
one sentence explaining the difference. Section 5.16.2's claim about explicit
separation becomes true of the whole report and is updated to point here.

An audit claim asserts that the figure quoted in the abstract equals the artifact
value, because the failure mode being guarded against is precisely an abstract
drifting out of sync with its source.

## Isolation

New module, new artifact directory. `h36m_crossview.py` is not modified and its
numbers do not change; the two are reported side by side.
