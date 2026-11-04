"""At what temperature does interchangeable counting change which arrangement is favoured? (T10)

    python scripts/naming_crossover.py

Prints to the screen and writes nothing: it is arithmetic, not a run.

THE TWO ARRANGEMENTS. Above a penalty of 1 the flat sheet is the lowest-energy arrangement and
the shattered state is an excited one, so named counting always prefers the sheet. Interchangeable
counting favours a shape by how symmetric it is (ASSUMPTION Q15), and the shattered state is far
more symmetric than the sheet. The two pull opposite ways, so there is a temperature where they
balance, and this works out where.

    flat sheet   S = N, X = 0            H = 0
    shattered    S = 1.5N, X = 2N        H = 16(N - 1.5N) + 4 lam (2N) = 8N (lam - 1)

    symmetry of the shattered state    (N/16) ln 192 + ln((N/16)!)     [Q15, measured]
    symmetry of the sheet              ln (2N)                          [translations x reflection]

The shattered state is favoured once  exp(-dH/g) * exp(d ln A)  exceeds 1, i.e. above

    g* = dH / d(ln A)

WHAT THIS IS NOT. A comparison of two specific arrangements, not of two full ensembles: it leaves
out the enormous number of disordered arrangements that dominate at high temperature. It says
which of these two the counting prefers, and that is all.
"""
from math import lgamma, log

LN192 = log(192.0)                      # renamings of one 4-cube, side-preserving (Q15, tested)


def energy_gap(n, lam):
    """H(shattered) - H(sheet), in the model's units."""
    return 8.0 * n * (lam - 1.0)


def symmetry_gap(n):
    """ln A(shattered) - ln A(sheet)."""
    k = n / 16.0
    return k * LN192 + lgamma(k + 1.0) - log(2.0 * n)


print("Crossover temperature g*, above which interchangeable counting prefers the shattered")
print("state and named counting still prefers the sheet. N = 160 unless stated.\n")

print("%-8s %12s %14s %12s" % ("penalty", "energy gap", "symmetry gap", "g*"))
for lam in (1.0, 1.05, 1.1, 1.25, 1.4, 1.5, 1.6, 2.0):
    n = 160
    d_h, d_a = energy_gap(n, lam), symmetry_gap(n)
    star = d_h / d_a
    print("%-8.2f %12.1f %14.2f %12s"
          % (lam, d_h, d_a, "every g" if d_h <= 0 else "%.2f" % star))

print("\nAnd how it moves with size, at a penalty of 1.25:\n")
print("%-8s %10s %12s %14s %10s" % ("points", "knots", "energy gap", "symmetry gap", "g*"))
for n in (64, 160, 320, 1600, 16000):
    d_h, d_a = energy_gap(n, 1.25), symmetry_gap(n)
    print("%-8d %10d %12.1f %14.2f %10.2f" % (n, n // 16, d_h, d_a, d_h / d_a))

print("""
Reading it. The energy gap grows in proportion to the number of points. The symmetry gap grows
faster, because k separate knots can be permuted among themselves in k! ways and that term
outruns any multiple of k. So g* FALLS as the system grows: the larger the network, the colder
it has to be before the sheet wins under interchangeable counting.

At a penalty of exactly 1 the two arrangements cost the same energy, so the symmetry gain is
unopposed and the shattered state is preferred at every temperature.""")
