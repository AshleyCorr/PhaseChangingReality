# Snap test: results on the real Planck sky

Run 16 November 2026 with the code and settings committed on 11 November (`snap_test.py`, `DIAL_SHEET.md`),
unchanged. Full numbers: `snap_test_run.json`. Every result is reported, including the ones against the models.
Ashley's work on his branch, not a registered test of Emily's hypothesis. Drafted with Claude (Anthropic).

## What the real sky shows

| Band (ℓ) | Lopsidedness | Percentile among 1,000 standard skies |
| --- | --- | --- |
| Large (2–40) | 0.132 | 97th (inside the middle 95%, just) |
| Medium (41–80) | 0.024 | 21st |
| Small (81–160) | 0.022 | 65th |

- The large-band tilt points toward Galactic (l, b) ≈ (208°, −17°), close to the direction Planck reports for its
  power asymmetry (to verify against the paper's table). This is a check that the method finds the known feature.
- **The direction is the same at every size.** Large vs medium: 1° apart (standard skies: median 86°, real sky at
  the 0th percentile). Large vs small: 13° apart (median 91°, 1st percentile). This is the one measurement outside
  the middle 95% of standard skies.
- **Patchiness is ordinary, slightly low** (8th to 26th percentile in the three bands). No excess patchiness.

## Verdicts, by the rules locked on 11 November

| Model | Fitted strength (large band) | Medium band | Small band | Direction agreement | Verdict |
| --- | --- | --- | --- | --- | --- |
| Standard | none | inside | inside | outside (too aligned) | the alignment across sizes is unusual |
| A: flawed crystal, gentle fade | A1 = 0.12 | **fails**: tilt 0.024, model predicts median 0.065 (2nd percentile) | inside (12th) | **fails** for large vs medium (real 1°, model median 33°); inside for large vs small | not supported as written |
| B: staggered self-observation, 60° patches, gentle fade | B1 = 0.02 (close to none) | inside | inside | **fails** (model median 82°) | not supported: no excess patchiness, no alignment |

## Reading (interpretation, ours, unverified)

- Neither model passes as written.
- The real sky's tilt keeps one direction across sizes, which is the kind of thing Model A's single built-in
  direction is about and Model B's random patches are not. But the tilt fades much faster than the gentle fade
  Ashley chose (about a fifth, not a half, from large to medium), and it is even more aligned than Model A's own
  fake skies.
- Planck's own papers report that the asymmetry's direction holds across scales while its strength falls (general
  knowledge, to verify), so this alignment is likely a known feature, not a new finding.
- Changing the fade to steep now would be fitting to what we have seen. A fair next test would lock a new rule
  first and check it on data not yet looked at: smaller ripples (ℓ 161 and up, which need a sharper map) or
  Planck's polarization maps.

## Caveats

- One sky. Fake skies have no foreground residuals or instrument noise; at these scales Planck's noise is small,
  but residual foregrounds outside the mask could add a common direction (not checked).
- The rehearsal planted only Model A; a planted Model B was not rehearsed.
