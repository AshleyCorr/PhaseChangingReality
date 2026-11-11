# Snap Test Dial Sheet

Locked 11 November 2026, approved by Ashley, before anything was measured on the real Planck sky. Living copy:
the Claude Doc "Snap Test Dial Sheet". This file is the date-stamped record; nothing below changes after the run.

This is Ashley's work, kept on his branch of his fork. It is not a registered test of Emily's hypothesis and
follows none of that repository's registration machinery. Drafted with Claude (Anthropic) in conversation with
Ashley.

It tests Ashley's two ideas for why the snap could be uneven: a **flawed crystal** (Model A) and **staggered
self-observation** (Model B).

## What Planck has already published

Planck already reports that the afterglow is slightly lopsided on the largest scales, so a model that just
reproduces a lopsided sky proves nothing. We know the answer before we start.

- **It's a large-scale effect.** The ripples are stronger on one half of the sky than the other, mostly on the
  biggest scales. Planck says removing the scales bigger than about ℓ = 100 removes the significance entirely.
- **It's modest.** A perfectly ordinary sky would show a tilt of about 3% just by chance. Planck finds more than
  that on the largest scales (exact values in the paper's Table 24, to verify), and treats it as a possible
  anomaly, not a discovery.

So the test rests on what the two models predict that nobody has fitted yet: **how the tilt changes from big
ripples to small ones**, **whether its direction stays the same across sizes**, and **how patchy the sky is**.

Source: Planck Collaboration, *Planck 2018 results VII, Isotropy and statistics of the CMB*, A&A 641, A7 (2020),
https://www.aanda.org/articles/aa/full_html/2020/09/aa35201-19/aa35201-19.html

## The data and the measurements

We measure three things at three ripple sizes, on the real sky and on 1,000 standard fake skies made the same way.

**The data**

- **Real sky:** Planck 2018 SMICA temperature map (`COM_CMB_IQU-smica_2048_R3.00_full.fits`, I_STOKES), shrunk to
  HEALPix Nside 128 and blurred to 1° (Gaussian FWHM), so only large patterns remain.
- **Galaxy mask:** Planck's common temperature mask (`COM_Mask_CMB-common-Mask-Int_2048_R3.00.fits`), degraded to
  Nside 128 and kept where above 0.9. It keeps 76% of the sky.
- **Standard fake skies:** 1,000 skies from the standard cosmological model with Planck 2018 settings, blurred
  and masked exactly like the real map.

**The ripple sizes**

| Band | Ripple scale (ℓ) | Roughly how big on the sky |
| --- | --- | --- |
| Large | 2–40 | bigger than about 5° |
| Medium | 41–80 | about 2° to 5° |
| Small | 81–160 | about 1° to 2° |

**The three measurements, in each band**

1. **Lopsidedness:** how much stronger the ripples are on one half of the sky than the other. Ripple strength is
   measured in 8° circles across the sky, and a simple two-halves tilt (a dipole) is fitted to it.
2. **Direction:** which way that tilt points, and whether the three bands point the same way.
3. **Patchiness:** how uneven ripple strength is from circle to circle, beyond the simple two-halves tilt.

## Model A: the flawed crystal

A flaw in the crystal gives the snap one built-in direction, so the sky's ripples are a little stronger toward
that side. In the boot-sequence picture the biggest structures set first and take the strongest imprint, so the
tilt fades as the ripples get smaller.

**How it's built:** each fake sky's ripples are multiplied by 1 + A × cos(angle from the flaw's direction), with A
set separately for each band.

| Dial | Setting | Where it comes from |
| --- | --- | --- |
| A1: tilt strength on the large band | Measured from the real sky, not predicted | The theory can't give a number for it yet |
| A2: fade from band to band | Half per smaller band (large 1 : medium ½ : small ¼) | Ashley's choice, 11 November, from the boot-sequence idea; a guess, not derived |
| A3: direction | Measured, not predicted | The theory doesn't say where the flaw points |

**Predictions:** the tilt points the same way in all three bands; it halves from large to medium and again from
medium to small (A1 is fitted on the large band only, the other two bands are held back to check the rule);
patchiness looks normal.

## Model B: staggered self-observation

Different regions of the crystal observe themselves, and snap, at slightly different moments, leaving patches of
stronger and weaker ripples with no single built-in direction.

**How it's built:** each fake sky's ripples are multiplied by 1 + B × g, where g is a random pattern of 60°
patches and B halves in each smaller band (large 1 : medium ½ : small ¼), the same gentle fade as Model A
(Ashley's choice, 11 November).

| Dial | Setting | Where it comes from |
| --- | --- | --- |
| B1: patch strength | Measured from the real sky, not predicted | The theory can't give a number for it yet |
| B2: patch size | 60°, chosen by Ashley before any look at the real sky | The theory doesn't set a size yet; one size fixed in advance |
| B3: fade | Half per smaller band, like Model A | Ashley's choice, 11 November |

**Predictions:** patchiness is higher than normal in all three bands; the tilt points wherever the biggest patches
happen to lean, the same way in all three bands; the tilt and the patchiness both halve from band to band. With
the same fade as Model A, patchiness is what tells the two models apart.

## How we decide

1. **Is the real sky unusual?** For each measurement, the real sky is placed among the 1,000 standard skies.
   Outside the middle 95% of them counts as unusual for the standard model.
2. **Fit on large, check on the rest.** Each model's strength is fitted on the large band only. Then 1,000 fake
   skies are made with that model and the medium and small bands are checked. A model passes a band if the real
   sky sits inside the middle 95% of that model's skies.
3. **Compare the models.** For each measurement, say which of A, B or standard puts the real sky nearest its
   middle.

Every result is reported, including the ones that go against the models. With only one real sky, a pass means
consistent with, never proven.

## Decisions (approved by Ashley, 11 November 2026)

- [x] Model A fade rule: half per smaller band (1 : ½ : ¼), over no fade or a steep fade (1 : ¼ : 1/16).
- [x] Model B patch size: 60° only. Model B fade: half per smaller band, like Model A.
- [x] Verdict rule: "unusual" means outside the middle 95% of fake skies.
- [x] Saving it: this sheet and the test code go to the branch `ashley/snap-test` in Ashley's fork. Nothing goes
  into Emily's repository.

## What this test can't tell us

- **We only have one sky.** A real sky that fits a model makes that story less of a fluke. It doesn't prove it.
- **Strengths and directions are fitted.** Only the fade rule, the patch size and the agreement between bands are
  real predictions.
- **It doesn't test the rest of the picture.** Black holes, the doughnut and the loop aren't touched.
- **It isn't a test of Emily's hypothesis.** This is Ashley's work, kept on his branch.
