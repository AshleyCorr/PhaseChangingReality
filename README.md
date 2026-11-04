# A phase-changing reality: the working copy

Emily Smith's hypothesis that our spacetime is one settled arrangement of something deeper, X, which was
stable for now and gave way in a sharp change, the burp, that released the energy of the Big Bang; and the
toy model on which its pieces are tested.

This repository holds **only what stands today**: the claim as it is now worded, the current definitions, the
tests and their verdicts, the code, and a list of what was tried and set aside. It was spun off on
4 November 2026 from the full record, `github.com/EmilySmithCreate/RecreationalPhysics`, which stays as the
archive: every decision with its date, every registered test with its reading, every result file, every paper
draft and every idea that did not survive. Nothing is deleted there.

| Read | For |
|---|---|
| `HYPOTHESIS.md` | the claim in the author's words, and the current definitions of X, the burp, curl, twist, particles, gravity, time, the loop |
| `WITHDRAWN.md` | ideas that were tested and removed, with what replaced each; not to be brought back |
| `STATE.md` | where each piece stands, what is running, what the owner is owed, what is next |
| `RESULTS.md` | every registered test, one line each, with its verdict and her prediction |
| `TOY.md` | the energy term by term with its source, and a map of the code |
| `GLOSSARY.md` | her words, the standard terms, and words not to use |
| `CLAUDE.md` | how an AI assistant works here: reading order, the rules, the things that come in opposites |
| `PREREGISTRATION.md` | tests registered in this repository from its creation on (the archive holds T1 to T102) |
| `ASSUMPTIONS.md` | assumptions and observation entries made here from its creation on (the archive holds Q1 to Q28 and O1 to O186) |

## The toy, in one paragraph

A network of points of two colours, each with four links (six in the three-direction version) joining black to
white; no two points share more than two neighbours. Its energy is a published model of emergent geometry
(Trugenberger's combinatorial quantum gravity: a reward for small square loops with a penalty where a link
carries too many) plus one cost of ours for curling tight, with strength λ − 1. Flat space is a torus of
squares. X's stand-in is a torus with one direction curled tight; it waits, then opens in one sharp change
and gives off exactly the curling cost per point. Stores beside the network hold energy so the books balance
to the last digit. On top of that sit states for curl and anti-curl on the squares, twist on the diagonals,
lengths on the links, and a lattice photon. At λ = 1 the toy is the published model; anywhere else it is our
family around it, never combinatorial quantum gravity.

## Status

A hobby project by a software engineer, not a physicist. No physicist has reviewed any of it; paper 1 is with
the published model's author. Nothing here is a claim about the real universe except where it is marked as the
author's hypothesis, stated to be argued with. Code, analysis and text were drafted with Claude (Anthropic),
directed and checked by the author.

## Commands

```bash
pip install -e ".[dev]"
pytest
```

The test suite was run in full when this repository was created and passed; see `STATE.md`.
