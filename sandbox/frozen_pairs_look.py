"""Sandbox look (9 November 2026, Ashley's idea, NOTEBOOK 9 November): matter and antimatter held side by side
in X, kept from meeting because time is frozen; flaws in the crystal leave some members unpaired; when time
starts, the pairs meet and cancel into light. Do the flaws give a consistent tilt to one kind, or random leftovers?

A stand-in, not the toy. A periodic L x L sheet, one square per site, each square holding one curl and one
anti-curl (the toy's wiring holds them paired, one of each per square). A flaw removes one member of a square's
pair: with probability q the anti-curl (leaving matter), otherwise the curl. Frozen: nothing moves. Unfrozen:
every leftover member takes one random step per tick, and a curl and an anti-curl on the same site cancel (into
light; not tracked). Cancelling conserves the net count, so the surplus is fixed the moment the flaws are made;
the run only shows where the leftovers end up.

Two kinds of crystal, the knobs fixed before running: flaw rate f = 0.05 of the squares; q = 0.5 ("random flaws",
nothing in the crystal prefers a hand) and q = 0.6 ("handed crystal", the crystal itself prefers one hand). Sides
32, 64, 128; 12 seeds each; 4000 ticks after unfreezing.

Predictions written before running:
  P1 Random flaws: a matter surplus in about half the crystals, an antimatter surplus in the rest; the surplus
     share |net| / squares falls as 1 / L (that is, 1 / sqrt(squares)), about sqrt(2 f / pi) / L:
     0.0056 at 32, 0.0028 at 64, 0.0014 at 128.
  P2 Handed crystal: a matter surplus in every crystal; share about f (2q - 1) = 0.010 at every side.
  P3 Random flaws leave both kinds alive in separate patches (the minority kind well above zero of the
     survivors at the end); the handed crystal leaves almost only matter.
Exact, not measured: P1's and P2's means follow from counting coins; the run checks the code and shows the spread.
"""
from pathlib import Path

import numpy as np

F = 0.05
SIDES = (32, 64, 128)
QS = (0.5, 0.6)
SEEDS = 12
TICKS = 4000
STEPS = np.array([[1, 0], [-1, 0], [0, 1], [0, -1]])


def cancel(pos, sign, L):
    """Remove equal numbers of + and - members sharing a site."""
    site = pos[:, 0] * L + pos[:, 1]
    plus = np.bincount(site[sign > 0], minlength=L * L)
    minus = np.bincount(site[sign < 0], minlength=L * L)
    k = np.minimum(plus, minus)
    if not k.any():
        return pos, sign
    keep = np.ones(len(sign), bool)
    for s in (1, -1):
        idx = np.flatnonzero(sign == s)
        order = idx[np.argsort(site[idx], kind="stable")]
        st = site[order]
        first = np.searchsorted(st, st, side="left")
        rank = np.arange(len(order)) - first
        keep[order[rank < k[st]]] = False
    return pos[keep], sign[keep]


def one(L, q, rep):
    rng = np.random.default_rng([20261109, L, int(q * 100), rep])
    flawed = np.flatnonzero(rng.random(L * L) < F)
    sign = np.where(rng.random(len(flawed)) < q, 1, -1)          # +1: anti-curl removed, curl (matter) left
    pos = np.stack([flawed // L, flawed % L], 1)
    net = int(sign.sum())
    made = len(sign)
    for _ in range(TICKS):
        if len(sign) == 0 or abs(int(sign.sum())) == len(sign):
            break
        pos = (pos + STEPS[rng.integers(0, 4, len(sign))]) % L
        pos, sign = cancel(pos, sign, L)
    assert int(sign.sum()) == net                                  # cancelling keeps the books
    left = len(sign)
    minority = int(min((sign > 0).sum(), (sign < 0).sum()))
    return dict(net=net, made=made, share=net / (L * L), left=left, minority=minority / left if left else 0.0)


def main():
    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    lines = [f"frozen_pairs_look  f={F}  seeds={SEEDS}  ticks={TICKS}",
             "q     side  matter-surplus  mean share   mean |share|  predicted |share|  survivors  minority share"]
    for q in QS:
        for L in SIDES:
            r = [one(L, q, k) for k in range(SEEDS)]
            share = np.array([x["share"] for x in r])
            pred = np.sqrt(2 * F / np.pi) / L if q == 0.5 else F * (2 * q - 1)
            lines.append(f"{q:<5} {L:<5} {int((share > 0).sum()):>3} of {SEEDS:<6}  {share.mean():+.5f}    "
                         f"{np.abs(share).mean():.5f}       {pred:.5f}            "
                         f"{np.mean([x['left'] for x in r]):>7.1f}    {np.mean([x['minority'] for x in r]):.3f}")
            print(lines[-1], flush=True)
    (out / "frozen_pairs_look_2026-11-09.txt").write_bytes(("\n".join(lines) + "\n").encode())


if __name__ == "__main__":
    main()
