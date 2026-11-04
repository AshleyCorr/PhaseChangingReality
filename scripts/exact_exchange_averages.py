"""EXACT: averages at N = 16 and 18 under three weightings of the points: named, interchangeable, and the exchange-sign
(fermionic) rule of VISION Update 29. Usage:

    python scripts/exact_exchange_averages.py [n ...]        # n = vertices a side: 8 (N = 16), 9 (N = 18)

Every valid state is enumerated by class (small_graphs.count_labelled_states and explore, task T4). Per named graph the
weight is exp(-H/g) times: 1 (named); |Aut| (interchangeable, VISION Update 12); the signed sum over the automorphisms,
which is |Aut| if every renaming is an even permutation and 0 if any is odd (the exchange rule). Summed over a class of
(n!)^2 / |Aut| named graphs, the class weighs, up to a common factor: (n!)^2 / |Aut| (named); 1 (interchangeable); 1 or 0
(exchange). So the exchange ensemble is the interchangeable one with the classes that have an odd renaming removed.
Printed: each class with its squares, surplus, symmetries and odd renamings; then phi = <S>/N and the folded share (X > 0)
under the three weightings at a grid of (g, lambda). Prints only; writes nothing.
"""
import sys
from math import exp, factorial
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity.cqg import NO_CAP, surplus, total_squares            # noqa: E402
from graphity.small_graphs import count_labelled_states, explore  # noqa: E402
from exact_exchange_sign import signed_sum                          # noqa: E402

GRID = [(g, lam) for lam in (0.0, 1.0, 1.25) for g in (1.0, 2.0, 5.0, 20.0)]


def classes_of(n):
    """[(squares, surplus, symmetries, odd renamings)] for every class at n vertices a side, no cap."""
    _, start = count_labelled_states(n, NO_CAP)
    if start is None:
        return []
    classes, _ = explore(start, NO_CAP)
    part = np.array([0] * n + [1] * n)
    out = []
    for adj, _ in classes.reps:
        a, odd, _ = signed_sum(adj, part)
        out.append((int(total_squares(adj)), int(surplus(adj)), a, odd))
    return out


def averages(cls, n, g, lam, weighting):
    big_n = 2 * n
    ws, ss, fs = [], [], []
    for s, x, a, odd in cls:
        boltz = exp(-(16 * (big_n - s) + 4 * lam * x) / g)
        if weighting == "named":
            w = factorial(n) ** 2 / a * boltz
        elif weighting == "interchangeable":
            w = boltz
        else:
            w = boltz if odd == 0 else 0.0
        ws.append(w); ss.append(s); fs.append(x > 0)
    tot = sum(ws)
    if tot == 0:
        return float("nan"), float("nan")
    return sum(w * s for w, s in zip(ws, ss)) / tot / big_n, sum(w for w, f in zip(ws, fs) if f) / tot


def main(ns):
    for n in ns:
        cls = classes_of(n)
        print("N = %d: %d classes; forbidden by the exchange sign (an odd renaming): %d"
              % (2 * n, len(cls), sum(1 for c in cls if c[3])))
        for k, (s, x, a, odd) in enumerate(sorted(cls, key=lambda c: (-c[0], c[1]))):
            print("  class %2d: S %2d X %2d symmetries %5d odd %4d -> %s" % (k, s, x, a, odd, "FORBIDDEN" if odd else "allowed"))
        print("  %5s %6s | %-24s | %-24s | %-24s" % ("g", "lambda", "named: phi, folded", "interchangeable", "exchange"))
        for g, lam in GRID:
            row = [averages(cls, n, g, lam, w) for w in ("named", "interchangeable", "exchange")]
            print("  %5g %6g | %9.4f %12.4f  | %9.4f %12.4f  | %9.4f %12.4f" % (g, lam, *row[0], *row[1], *row[2]))


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [8, 9])
