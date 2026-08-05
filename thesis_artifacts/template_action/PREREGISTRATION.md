# Pre-registration: does the template baseline have a failure mode?

Written and committed **before** the run. Ninth pre-registration.

Date: 2026-08-06

## The question

Section 5.10.1 reports the template baseline beating the anatomical frame on all
180 pairs, and Section 5.10.3 shows that holds under three templates and every
centring. "All 180 pairs" is nonetheless 180 pairs drawn from fifteen actions
that are largely upright, and the template is a single fixed skeleton close to a
standing pose.

A method that aligns every prediction onto one reference ought to struggle where
the pose is far from that reference. If it does not, the anatomical frame has no
regime of its own and the report should say so plainly rather than leave the
reader to wonder. If it does, that regime is worth naming.

## Test

Break the same comparison down by action, both backbones, scored on the thirteen
non-constructor joints as the headline is. Report per action: the anatomical
distance, the template distance, and the difference.

Human3.6M's fifteen actions include several far from standing --- SittingDown,
Sitting, Photo --- and several close to it --- Walking, WalkTogether, Directions,
Posing.

## Readings, fixed now

1. **The template wins on every action.** The anatomical frame has no regime in
   which it is preferable on these data. Section 5.10.1 says exactly that, and
   the report stops implying a niche exists.
2. **The template loses on some actions, and they are the ones far from a
   standing pose.** That is a real failure mode with a mechanism, and it names
   the regime where an anatomically-defined frame is preferable.
3. **The template loses on some actions with no such pattern.** Report the
   actions and state that we have no mechanism for it, rather than inventing one.

## What is not claimed

A per-action breakdown of 12 pairs each is a weaker instrument than the pooled
comparison, and no interval is computed per action. Any pattern found here is
descriptive and is reported as such.

## Isolation

Reuses `evaluation/template_baseline.py`. No audited file changes.
