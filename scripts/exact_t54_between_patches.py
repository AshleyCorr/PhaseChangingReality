"""EXPLORATORY, exact (11 October 2026): the energies BETWEEN two completed patches of T54. Usage:

    python scripts/exact_t54_between_patches.py [L]

Why. T54 prices completed square patches (k x k columns opened) and fits dH(k) = a k - b k^2. An outside review of
11 October pointed out that going from the k x k patch to the (k + 1) x (k + 1) patch takes 2k + 1 switches, that
the states in between were never priced, and that the fitted peak therefore bounds nothing. This script prices them:
every order of the 2k + 1 column switches from patch k to patch k + 1, with the highest energy met on the best
order. It is still only the prescribed switches of T54's construction: other switches, other patches or detours
could go lower, so the number is an upper bound on the barrier ALONG THIS FAMILY OF PATHS and not the barrier.

Nothing here is a registered rule and no verdict changes. It reads and writes nothing in results/.
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity.cqg_d import is_valid, torus                        # noqa: E402
from exact_t54_patch import energy, helix_switch, patch           # noqa: E402

CASES = (("all_at_the_last", 1.10), ("all_at_the_last", 1.15), ("all_at_the_last", 1.20), ("all_at_the_last", 1.25),
         ("all_at_the_last", 1.40), ("none", 1.10), ("none", 1.25), ("none", 1.40))


def slab(length):
    return torus([4, length, length])[0]


def new_columns(k):
    """The 2k + 1 columns that turn the k x k patch into the (k + 1) x (k + 1) patch."""
    return [(x, k) for x in range(k)] + [(k, y) for y in range(k)] + [(k, k)]


def climb(length, k, lam, tie, limit=5040):
    """(energy of patch k, energy of patch k + 1, the lowest peak over the orders tried, that order's energies),
    all above the untouched slab. Every order is tried while there are at most `limit` of them; beyond that, the
    orders that fill the new row and the new column outward from one end and close with the corner."""
    base_adj = slab(length)
    base = energy(base_adj, lam, tie)
    start, _ = patch(base_adj, length, k)
    cols = new_columns(k)
    orders = itertools.permutations(cols)
    if k >= 4:                                                        # 9! orders and more: a fixed sample
        row, col = cols[:k], cols[k:2 * k]
        orders = [row + col + [cols[-1]], col + row + [cols[-1]], row[::-1] + col[::-1] + [cols[-1]],
                  [c for pair in zip(row, col) for c in pair] + [cols[-1]]]
    tried = []
    for order in orders:
        adj = start.copy()
        seen = [energy(adj, lam, tie) - base]
        for x, y in order:
            helix_switch(adj, length, x, y)
            seen.append(energy(adj, lam, tie) - base)
        tried.append((max(seen), seen, order))
    for _peak, seen, order in sorted(tried, key=lambda item: item[0]):    # the lowest peak whose states are all valid
        adj = start.copy()
        if all(is_valid(adj) and (helix_switch(adj, length, x, y) or True) for x, y in order) and is_valid(adj):
            return seen[0], seen[-1], max(seen), seen
    raise ValueError("no order of the prescribed switches passes through valid graphs only")


def row(length, lam, tie, longest=6, along_y=True):
    """Energies above the slab of a single line of 1, 2, ... columns opened: not squares. The construction is not
    the same in its two directions, so the line can run along y (the cheaper way) or along x. Each state valid."""
    base_adj = slab(length)
    base = energy(base_adj, lam, tie)
    adj, out = base_adj.copy(), []
    for i in range(longest):
        helix_switch(adj, length, 0 if along_y else i, i if along_y else 0)
        assert is_valid(adj)
        out.append(energy(adj, lam, tie) - base)
    return out


def main(length=16, ks=(1, 2, 3)):
    print("T54's construction, the states between completed patches (exact; slab 4 x %d x %d)." % (length, length))
    print("For each step k -> k + 1: the two completed-patch energies, and the highest energy on the best order of")
    print("the 2k + 1 prescribed switches. All above the untouched slab.\n")
    for tie, lam in CASES:
        for k in ks:
            e0, e1, peak, path = climb(length, k, lam, tie)
            print("%-15s lambda=%.2f  patch %d -> %d: %7.1f -> %7.1f | highest on the way %7.1f | %s"
                  % ("tied" if tie != "none" else "untied", lam, k, k + 1, e0, e1, peak,
                     " ".join("%.0f" % e for e in path)))
    print()
    print("A single line of columns, not a square (1, 2, ... columns), above the untouched slab; along y | along x:")
    for tie, lam in CASES:
        print("%-15s lambda=%.2f  %s | %s" % (
            "tied" if tie != "none" else "untied", lam, " ".join("%.1f" % e for e in row(length, lam, tie)),
            " ".join("%.1f" % e for e in row(length, lam, tie, along_y=False))))
    return 0


if __name__ == "__main__":
    sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 16))
