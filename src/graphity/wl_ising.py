"""The two-dimensional Ising model, used only to check the Wang-Landau schedule (task T6).

WHY IT IS HERE. Task T6 asks that the flat-histogram sampler be validated on a system
whose answer is known before it is pointed at the graph model. The Ising model on a small
square lattice is the standard choice because its density of states can be counted by brute
force: at L = 4 there are 2^16 = 65 536 states, and a computer can simply look at all of
them. `exact_dos` does exactly that, with no cleverness at all, so it cannot be wrong in the
same way the sampler is.

WHAT THIS CHECKS AND WHAT IT DOES NOT. `wang_landau.anneal` holds the schedule: when to halve
ln f, when to hand over to the 1/t rule, when to stop. That schedule is shared, so a check
here is a real check of it. The *graph* kernel is not exercised here at all; that is checked
separately against our own exhaustive enumeration at N = 16 and 18. Keeping the two apart is
the point: if one fails we know which.

THE MODEL. Spins s = +/-1 on an L x L lattice with periodic edges, E = - sum over neighbouring
pairs of s_i s_j, each pair counted once. E runs from -2 L^2 to +2 L^2 in steps of 4, so bin
k holds E = 4 k - 2 L^2. Two of those bins are empty at every L (E = -2L^2 + 4 cannot be
made), which is useful: it means the bookkeeping for unreachable bins gets exercised here too.
"""
import numpy as np
from numba import njit

from graphity.wang_landau import both_stages


def exact_dos(side):
    """Density of states by looking at every configuration. Returns (energies, counts).

    Only sensible up to about L = 5 (2^25 states). Nothing here is shared with the sampler.
    """
    n = side * side
    if n > 25:
        raise ValueError("brute force only; use side <= 5")
    right = np.array([(i // side) * side + (i % side + 1) % side for i in range(n)])
    down = np.array([((i // side + 1) % side) * side + i % side for i in range(n)])
    counts = np.zeros(n + 1, dtype=np.int64)
    for state in range(1 << n):
        spins = np.array([1 if state >> i & 1 else -1 for i in range(n)], dtype=np.int64)
        energy = -int((spins * spins[right]).sum() + (spins * spins[down]).sum())
        counts[(energy + 2 * n) // 4] += 1
    energies = 4 * np.arange(n + 1) - 2 * n
    return energies, counts


@njit(cache=True)
def _ising_sweeps(spins, right, down, left, up, lng, hist, seen, ln_f, n_sweeps, seed, state, floor):
    """Single-spin-flip Wang-Landau sweeps. state = [bin index], read and written in place.

    seed < 0 carries on the previous call's random stream; see `wang_landau._wl_sweeps`.
    """
    if seed >= 0:
        np.random.seed(seed)
    n = spins.shape[0]
    k = state[0]
    attempted, accepted = 0, 0
    for _ in range(n_sweeps):
        for _ in range(n):
            attempted += 1
            i = np.random.randint(0, n)
            local = spins[right[i]] + spins[down[i]] + spins[left[i]] + spins[up[i]]
            d_e = 2 * spins[i] * local              # E = -sum s_i s_j, so flipping i costs this
            j = k + d_e // 4
            ok = 0 <= j < lng.shape[0]
            if ok:
                there = lng[j, 0] if seen[j, 0] else floor
                d_l = lng[k, 0] - there
                if d_l < 0.0 and np.random.random() >= np.exp(d_l):
                    ok = False
            if ok:
                spins[i] = -spins[i]
                k = j
                accepted += 1
            if not seen[k, 0]:
                seen[k, 0] = True
                lng[k, 0] = floor
            lng[k, 0] += ln_f
            hist[k, 0] += 1
    state[0] = k
    return attempted, accepted


def dos_ising(side, **kw):
    """Density of states of the L x L Ising model. Returns the dict of `both_stages`."""
    n = side * side
    idx = np.arange(n)
    right = (idx // side) * side + (idx % side + 1) % side
    left = (idx // side) * side + (idx % side - 1) % side
    down = ((idx // side + 1) % side) * side + idx % side
    up = ((idx // side - 1) % side) * side + idx % side
    spins = np.ones(n, dtype=np.int64)
    state = np.array([0], dtype=np.int64)           # all up: E = -2n, bin 0

    def advance(lng, hist, seen, ln_f, n_sweeps, seed):
        floor = float(lng[seen].min()) if seen.any() else 0.0
        return _ising_sweeps(spins, right, down, left, up, lng, hist, seen,
                             ln_f, n_sweeps, seed, state, floor)

    return both_stages(advance, (n + 1, 1), **kw)
