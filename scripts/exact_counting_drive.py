"""EXACT, exploratory: how the counting drive to curl depends on the number of points (the owner's question of
2026-09-26, VISION Update 33: how many micro degrees of freedom set the symmetry gained by curling, and do the three
interchangeable large directions contribute on their own?). Usage:

    python scripts/exact_counting_drive.py

With interchangeable points an arrangement's weight carries its number of side-preserving renamings A, so its free
energy is H - g ln A (ASSUMPTIONS O55). Compared here: flat space, an L^D torus (A = (N/2) 2^D D!: the translations that
keep the two sides, times the symmetries of the cubic lattice, of which D! are the permutations of the D directions),
against the fully curled state, a gas of k separate cubes Q_{2D} (each with (2D)! 2^(2D) / 2 side-preserving
renamings, and k! for exchanging identical cubes), costing D * 4(lambda - 1) per point above flat.

The two cross where D 4(lambda - 1) N = g [k ln A_cube + ln k! - ln A_flat], N = k 2^(2D). O55 computed this at N = 512 and
said the crossing does not move with N; that is wrong once ln k! is kept (it is in O55's own count): ln k!/k grows as
ln k - 1, so the crossing lambda rises with the logarithm of the number of cubes. Checked against O55 at N = 512
(1.020 at g = 1.5, 1.013 at g = 1.0) by tests/test_exact_counting_drive.py.

Ours; arithmetic on exact counts. The flat torus and the gas are compared as single arrangements: neither's thermal
excitations are counted.
"""
from math import factorial, lgamma, log


def ln_a_cube(dim):
    """ln of the side-preserving renamings of the hypercube Q_(2 dim)."""
    m = 2 * dim
    return log(factorial(m) * 2 ** m / 2)


def ln_a_flat(n, dim):
    return log(n / 2 * 2 ** dim * factorial(dim))


def gap_per_point(n, dim, g):
    """g [ln A_gas - ln A_flat] / N: the counting drive to curl, per point."""
    k = n / 2 ** (2 * dim)
    return g * (k * ln_a_cube(dim) + lgamma(k + 1) - ln_a_flat(n, dim)) / n


def crossing_lambda(n, dim, g):
    """The lambda below which the fully curled gas has the lower free energy."""
    return 1.0 + gap_per_point(n, dim, g) / (4.0 * dim)


def crossing_lambda_ln(ln_k, dim, g):
    """crossing_lambda with the number of cubes given as ln k (Stirling for ln k!, exact to 1/(12k))."""
    k_cells = 2 ** (2 * dim)
    ln_n = ln_k + log(k_cells)
    ln_fact_per_cube = ln_k - 1.0 + 0.5 * (log(6.283185307179586) + ln_k) / 2.718281828459045 ** min(ln_k, 700)
    ln_flat = ln_n - log(2) + dim * log(2) + log(factorial(dim))
    gap = g * (ln_a_cube(dim) + ln_fact_per_cube - ln_flat / 2.718281828459045 ** min(ln_k, 700)) / k_cells
    return 1.0 + gap / (4.0 * dim)


def cubes_for(lam, dim, g):
    """log10 of the number of cubes at which the counting drive beats the curling cost at lambda (bisection)."""
    lo, hi = 3.0, 5000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if crossing_lambda_ln(mid, dim, g) < lam:
            lo = mid
        else:
            hi = mid
    return lo / log(10)


def main():
    print("per point, from the cube alone (ln A_cube / 2^(2D)): D=2 %.4f  D=3 %.4f  D=4 %.4f"
          % tuple(ln_a_cube(d) / 2 ** (2 * d) for d in (2, 3, 4)))
    print("the D! permutations of the large directions: ln D! = %.2f, %.2f, %.2f in total, on both sides"
          % tuple(log(factorial(d)) for d in (2, 3, 4)))
    for dim in (2, 3, 4):
        for g in (1.0, 1.5):
            row = []
            for log10k in (0.903, 3, 6, 12, 24, 60):
                row.append("k=10^%-5g lambda*=%.4f" % (log10k, crossing_lambda_ln(log10k * log(10), dim, g)))
            print("D=%d g=%.1f  %s" % (dim, g, " | ".join(row)))
        print("D=%d: cubes needed for counting to beat the curling cost: lambda=1.10 k=10^%.1f, 1.25 k=10^%.1f (g=1.5)"
              % (dim, cubes_for(1.10, dim, 1.5), cubes_for(1.25, dim, 1.5)))


if __name__ == "__main__":
    main()
