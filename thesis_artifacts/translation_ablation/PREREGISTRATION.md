# Pre-registration: is the baseline's advantage orientation or translation?

Written and committed **before** the run. Eighth pre-registration.

Date: 2026-08-06

## The confound

Section 5.10.1 reports the template baseline beating the anatomical frame, and
states with the number that the two methods normalise translation differently.
The anatomical frame is root-centred: every pose has joint 0 at the origin. The
template baseline is centroid-centred, because `procrustes_align` subtracts the
mean of the point set.

Centroid-centring averages seventeen joints where root-centring relies on one,
so it is more robust to a noisy prediction at that one joint. Part of the
baseline's advantage may therefore be better translation handling rather than
better orientation, and the comparison as run cannot distinguish them.

## The design

A two-by-two, on the same 180 pairs and both backbones. Only the centring of the
final output changes; neither the frame construction nor the Kabsch fit is
touched, and the fit is translation-invariant in any case.

|                  | root-centred            | centroid-centred        |
|------------------|-------------------------|-------------------------|
| anatomical frame | as reported, 75.3 mm    | new                     |
| template         | new                     | as reported, 53.0 mm    |

## Readings, fixed before the numbers exist

Let $G$ be the reported gap, 22.3 mm on the first backbone and 8.8 on the second,
and let $G'$ be the gap with both methods centred the same way, averaged over the
two centrings.

1. **$G' \approx G$.** Translation normalisation is not the story; the
   baseline's advantage is its rotation estimate. Section 5.10.1's caveat is
   discharged and the sentence saying we have not separated the two is replaced
   by the measurement.
2. **$G'$ materially smaller than $G$.** A substantial part of the reported
   advantage is translation handling rather than orientation. Section 5.10.1
   must say what fraction, and the headline comparison should be restated
   between like-centred methods.
3. **$G'$ larger than $G$.** Root-centring was flattering the baseline; the
   orientation advantage is larger than reported. Say so.

No prediction is made about which fires. Centroid-centring plausibly helps both
methods, and whether it helps them by similar amounts is the open question — if
it did not, the anatomical frame would have to be losing translation accuracy
specifically, which nothing in this report suggests.

## Isolation

Reuses `evaluation/template_ablation.py`. No audited file changes and no
recomputation of any existing number.
