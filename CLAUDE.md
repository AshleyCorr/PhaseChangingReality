# Project memory for Claude Code

This repository is the **working copy** of Emily Smith's hypothesis about emergent spacetime and of the toy
model it is tested on. It holds only what stands today. The full record, with every decision, every test and
every idea that was tried and set aside, is the archive repository
`github.com/EmilySmithCreate/RecreationalPhysics` (the branch `claude/particle-sector-2026-10-24` as of
4 November 2026). Created 4 November 2026 at the owner's request, because an assistant reading the full
archive kept bringing back ideas that had been tested and removed, and kept making statements the record had
already corrected.

## Read in this order, every session

1. This file.
2. `HYPOTHESIS.md`: the claim as it stands, in her words, and the current definitions of every object.
3. `WITHDRAWN.md`: what was tried and set aside. **Never propose, assume or "remind" her of anything on it.**
   If you think a withdrawn idea should come back, say so once, as a question, with its line from that file.
4. `STATE.md`: where each piece stands, what is running, what is owed, what is next.
5. `RESULTS.md`: every registered test and its verdict, one line each.
6. `TOY.md`: the toy term by term, and the code map.
7. `GLOSSARY.md`: her words, the standard terms, and words not to use.

The archive is for looking up a detail (a number, a quotation, a derivation). Do not re-read it to decide what
the hypothesis is; that is what `HYPOTHESIS.md` and `WITHDRAWN.md` are for. If this repository and the archive
disagree, this repository is newer unless the archive's commit is later than the date at the top of `STATE.md`,
in which case say so and bring this repository up to date.

## Who the owner is and how she works

Emily is a software engineer, not a physicist. The hypothesis is hers; the toy, the tests and the writing are
drafted with Claude in conversation with her. Her aim: a sound model that can make a prediction that was not
put in and could be checked against the sky. Her messages are a stream of ideas from intuition and from
lectures watched years ago; writing them down is a translation from pictures. She needs them **checked and
corrected quickly**, not held to as claims, and not flattered. Every reply to such a message leads with three
things: what checks out, what does not (plainly, with the published fact), and the nearest published model.

She does not need novelty. "This is already published" is good news to her: published mathematics is what she
wants to apply to her picture for predictions and constants. Never headline "nothing new here".

When she gives an idea, capture it first in her words (the notebook, `sandbox/NOTEBOOK.md`), run the cheapest
exact look if one exists (most take seconds), then judge. Label every objection as one of three kinds before
making it: a *convention* (free to change, never a reason to resist), a *measurement* (her rules must
reproduce the number), or a *theory's structure* (replaceable by anything that gives the same numbers). Only a
measurement may stop an idea, and only after the arithmetic has been done.

## Things that come in two opposite kinds (do not say otherwise)

The toy and the hypothesis both hold several things that come in two opposite kinds. An assistant has wrongly
written "nothing here comes in opposites" more than once. The list, so that it is never written again:

- **Point colour**: every point is black or white; a link joins black to white (the wiring is bipartite).
- **Curl and anti-curl**: on every square the black-to-black diagonal carries curl (matter) and the
  white-to-white diagonal carries anti-curl (antimatter). The wiring always holds them paired, one of each
  per square; the *energy* cannot tell them apart; a surplus of one kind lives in the state on the squares
  (`graphity.curl_pairs`). Say "the energy cannot tell the two kinds apart", never "the toy has nothing in
  opposite kinds".
- **Twist handedness**: twist is signed; the sign is the colour of the diagonal (black +, white −); a diagonal
  runs black to black, an anti-diagonal white to white.
- **Matter and antimatter**: the two colours of twist; made as equal pairs by rule (one to one); a black and
  a white twist that meet cancel into two photons.
- **The twin daughters** of one burp: one with a small matter surplus, one with the matching antimatter
  surplus; the black hole stays neutral.
- **The two curled slabs of X**: one curled right-handed, one left-handed, a rod between them.

## Non-negotiable rules (the owner's; the full text with dates is in the archive's `CLAUDE.md`)

1. **Every assumption has a source or is marked "Ours".** Status Sourced / Calibrated / Ours / Unverified,
   with section or equation numbers. Never invent a citation; if unsure, write "to verify".
2. **Reproduce before extending.** A published number is matched before any result on that code is interpreted.
3. **No cherry-picking.** Knobs fixed in advance; every setting run is reported.
4. **Pre-register.** Predictions and analysis choices are written in `PREREGISTRATION.md` before the runs,
   with her prediction recorded in her words (or marked *inferred* until she confirms it). Exploratory runs say
   so in their config's `_purpose`.
5. **Results are append-only.** Every result comes from a config in `configs/` through a script in `scripts/`,
   with seeds in the config and a `.meta.json` beside the CSV; the runners refuse to overwrite; only a
   `*.partial` file may be deleted. The archive's `results/` are not copied here (three thousand files); cite
   them by path in the archive.
6. **Tests pass** (`pytest`) before every commit. A new kernel needs a brute-force check (networkx), an
   exactness check of incremental updates, constraint-preservation checks and a same-seed check.
7. **Say what is not known.** Exact, measured, published, guessed: say which. AI physics reasoning is
   "ours, unverified". A first look is not a finding. Inconvenient results are reported as prominently as
   convenient ones.
8. **Plain language for the owner.** Define terms, lead with the answer, push back when something does not hold.
9. **Ideas start in the notebook** (`sandbox/NOTEBOOK.md`, hers verbatim with the date) and may be tried in
   the sandbox at once. `HYPOTHESIS.md` changes only for a decision of hers; the commit says why.
10. **Four kinds of statement, never silently promoted**: exact; measured; interpretation (ours, unverified);
    her claim about reality. Anything an outside reader sees says which kind each statement is.
11. **The cheapest decisive test first.** When a registered prediction fails: her ideas captured first, in her
    words; before a replacement is promoted, write what exact test tells the causes apart and whether it has
    run. A failed prediction keeps the form "previous claim → failed because → replacement → new test".
12. **A red-team pass at each milestone**, by a fresh read-only session (a different model where available),
    given the question and never the conclusions; recorded with what it found and what it got wrong.
13. **Outside theories are targets, not confirmations, and a close number is not a result.** A number counts
    only after it was derived independently, its input uncertainty propagated, its inputs checked for containing
    the target, its neighbours tested, and the other arbitrary combinations that would match counted.
14. **A held-out size for every claim about size**: fit on the smaller sizes, write a number with a range for
    the largest before it runs, score the size claim there alone.
15. **A written prior-work search behind every "we have not found"** (`docs/reading/notes/`, template there).
    Never write "nobody has done this".
16. **The sandbox** (`sandbox/`) is for looking: no registration, no labels, one line in `sandbox/LOG.md`. It can
    kill but not crown; it does not peek at unread registered results; promotion needs fresh data.
17. **One fact, one home.** A result is written once in full (its registration's "Reading" and one O entry in
    `ASSUMPTIONS.md`); everything else links to it. Prose gives the date; commit hashes give the order.
18. **A review of the rules each month** (`docs/RULES_REVIEW.md`).
19. **Exact look before memory paragraph.** A cheap exact test runs before any "what it must face" is written;
    a physics statement from memory is marked "general knowledge, to verify" and gets a source within the week.
20. **One object at a time.** A design brief keeps a "current definitions" box at its head; a definition changes
    only after its named exact test has run, or by her withdrawal; her answers are read back as one set.
21. **Every question she asks is answered in the record**, in a dated line, not only in chat.
22. **A fresh reader audits any day with more than three decisions.**
23. **Every objection names its kind** (convention, measurement, theory structure; see above).

Naming: at λ ≠ 1 the toy is **our family around the published model**, never "combinatorial quantum gravity"
(the model's author's condition). Public pages say "the published model" and "the toy", never a bare "the
model". American spelling from 24 September 2026 on, with one exception: "programme", her joke, never
"program".

## Working habits that have bitten

- `git fetch` before recording anything. Commit with `git add <paths>` and `git commit -o <paths>`, never
  `git add -A` (other sessions share the index). Check the GitHub tests workflow after every push.
- Run `date -u` before writing a date. Prose carries the date, not the clock time.
- Bash heredocs eat backslashes on this laptop: write scripts and long text with the Write tool.
  `grep -c` returning 0 exits 1 and breaks `&&` chains. `Path.write_text` rewrites LF as CRLF: pass
  `newline="\n"`, read with `read_bytes().decode()`, and check `git diff --stat` before committing.
- Python: the Microsoft Store Python has the packages
  (`/c/Users/emily/AppData/Local/Microsoft/WindowsApps/python.exe`); set `PYTHONIOENCODING=utf-8` when
  printing non-ASCII.
- Cloud runs (AWS Batch, up to 96 at once) are submitted by pushing a manifest to `cloud/queue/` **in the
  archive repository**; that machinery was not copied here. The laptop's AWS profile is the wrong account.
- Read positions from the wiring before naming a shape. Five times a shape was named from a count and was wrong.
- A threshold per point is an absolute threshold proportional to N; check whether a criterion itself scales.
- The owner's correspondence with physicists is private and local (`docs/outreach/` in the archive,
  gitignored). Never quote or closely paraphrase an email in a tracked file.

## Commands

```bash
pip install -e ".[dev]"
pytest                       # all tests (several minutes)
python scripts/run_cqg_sweep.py configs/cqg_first_look.json
```

## Disclosure

All code and documents were drafted with Claude (Anthropic) in conversation with the owner. Keep the
disclosure in `README.md` accurate.
