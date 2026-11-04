"""Density of states by Wang-Landau flat-histogram sampling (task T6; [WL01]).

WHY THIS AND NOT ANOTHER LONG RUN. Every measurement in this repository so far is
an average taken at one coupling: the chain sits where the energy is comfortable
and reports what it sees. That can never show a barrier, because a barrier is
exactly the place a thermal chain refuses to go. Wang-Landau turns the problem
round. Instead of sampling states in proportion to exp(-H/g), it builds its own
weights until it walks *flat* over the energy: it is then forced through the
barrier, and the weights it needed are the density of states. From the density of
states every temperature follows at once, and so does the shape S2 asks for.

WHAT IS COUNTED, AND WHY IT IS (S, X) AND NOT THE ENERGY. The energy of this model
is H = 16 (N - S) + 4 lambda X, so a walk over the pair (S, X) is a walk over the
energy of *every* lambda at once. One run gives the whole knob map, which is what
VISION S1 asks for (publish every setting, not a chosen one), and it means the
answer does not have to be recomputed when lambda changes. The price is a
two-dimensional histogram instead of a one-dimensional one.

THE CONVERSION, once ln g(S, X) is known:

    Z(g, lambda)     = sum over bins of exp(ln g - H/g)
    <A>(g, lambda)   = sum A exp(ln g - H/g) / Z
    F(g, lambda; H)  = H - g ln g(S, X)      the free energy of one bin

and the canonical energy histogram of VISION S2 is P(H) proportional to
exp(ln g(S, X) - H / g), summed over the bins at that H.

WHAT THE NUMBERS MEAN. ln g is only ever known up to one additive constant, which
cancels in every average above. `normalise_to` fixes it against a bin whose exact
count is known, which is how the small-size check below is made.

HOW IT IS DRIVEN. `anneal` is model-agnostic: it holds the schedule (flatness,
halving, and the 1/t rule of [BP07]) and knows nothing about graphs. It is given
an `advance` function that runs sweeps of some model and reports back. Two such
functions exist: `graph_sweeper` here, and the Ising one in `wl_ising.py`, whose
answer is known exactly by brute force. So the schedule is checked against a
textbook system and the graph kernel is checked against our own exhaustive
enumeration at N = 16 and 18 (`results/ergodicity_small.csv`), and a failure can
be told apart.

TWO STAGES, AND WHY THE SECOND ONE IS NOT OPTIONAL. Wang-Landau on its own does not
converge to a useful precision, and this is not a defect of our code: the error is
known to saturate, so that running longer stops helping [BP07]. We measured it on
the Ising model, where the answer is known exactly: ten times the work left the
error where it was, at 20 to 30 % in g. The 1/t rule of [BP07] improves it but does
not fix it. The cause is visible in the numbers: a correction round is only as good
as the walk's evenness inside that round, and a walk that has not crossed the range
many times is not even.

So `anneal` is only stage one, and its job is to find the *shape*. Stage two is
`refine`, and it is exact in a way stage one cannot be. Freeze the weights, stop
updating them, and simply count visits: that is ordinary importance sampling with
known weights, so the count of visits to a bin is proportional to g times the weight,
with no schedule and nothing to tune. Adding ln(visits) to the frozen ln g gives the
answer, and its error is plain statistics that falls like one over the square root
of the time. Iterating it converges. On the Ising model this takes 20 % to under
1 % in one pass. **Report nothing from stage one alone.**

`one_over_t=False` gives plain halving instead, worth running once as a control.

WHAT THIS MODULE DOES NOT DO. It does not decide anything. Whether a dent in the
entropy curve counts as a first-order transition is a question with a written
answer in `PREREGISTRATION.md`, fixed before the production runs.

TRUSTING A RESULT. One run is never enough: the driver is deterministic given its
seed, so run several seeds and compare. `spread` reports the largest difference
between independent runs after each is shifted to a common reference bin, which is
the error bar that matters.
"""
import warnings

import numpy as np
from numba import njit

from graphity.cqg import (CAP, ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, NO_CAP, _new_edge_ok, _note_square_edges,
                          _replace, _surplus_on, _switch, has_edge, squares_on_edge, surplus, total_squares)


def anneal(advance, shape, seed=0, ln_f_final=1e-6, flat=0.8, sweeps_per_check=50,
           max_moves=2 * 10 ** 9, one_over_t=True, max_rounds_per_step=40):
    """Run the Wang-Landau schedule until ln f falls below `ln_f_final` or the move budget runs out.

    advance(lng, hist, seen, ln_f, n_sweeps, seed) -> (moves, accepted) runs that many
    sweeps of some model, adding ln_f to lng and 1 to hist at the current bin after every
    attempted move, and setting seen where it has been. It is free to modify all three.

    shape  : the bin grid, 2-D. A one-dimensional model uses (n, 1).
    flat   : a round ends when the least-visited seen bin has at least this share of the
             mean over seen bins. Ignored once the 1/t rule has taken over.

    max_rounds_per_step : halve ln f after this many rounds even if the histogram is not flat.
             Not a tuning knob but a guarantee. Measured at N = 36 with 738 bins: without it
             the flatness target was never reached, ln f never left its starting value of 1,
             the accumulated weights ran to a spread of 25 000, and the frozen-weight stage
             then could not move at all (one bin visited). Stage one only has to find the
             SHAPE of ln g; `refine` is what makes it accurate.

    Returns a dict: lng, seen, rounds (one row each, for the report), moves, ln_f, accept,
    `discovery_round` (the last round in which a new bin was found) and `stalled` (True if the
    move budget ran out before ln f reached its target). Look at `stalled` and at ln_f first:
    a run that ends with ln_f still near 1 has not measured anything.
    """
    lng = np.zeros(shape, dtype=np.float64)
    hist = np.zeros(shape, dtype=np.int64)
    seen = np.zeros(shape, dtype=np.bool_)
    ln_f, moves, accepted, rounds = 1.0, 0, 0, []
    n_seen, discovery_round, tail, since_step = 0, 0, False, 0
    first = int(np.random.SeedSequence(seed).generate_state(1)[0])
    while ln_f > ln_f_final and moves < max_moves:
        sub = first if not rounds else -1      # seed once, then carry the stream on (Q14)
        did, ok = advance(lng, hist, seen, ln_f, sweeps_per_check, sub)
        moves += did
        accepted += ok
        grew = int(seen.sum()) > n_seen
        n_seen = int(seen.sum())
        since_step += 1
        if grew:
            # While new bins are still turning up, flatness is meaningless: every bin found
            # late has almost no counts and drags the minimum down, so the criterion can never
            # be met and ln f would never fall. Start the window again instead.
            discovery_round = len(rounds)
            hist[:] = 0
            since_step = 0
        counts = hist[seen]
        flatness = counts.min() / counts.mean() if counts.size and counts.mean() > 0 else 0.0
        if tail:
            ln_f = n_seen / moves                      # the 1/t rule: t = moves per bin
        elif one_over_t and ln_f < n_seen / moves:
            tail = True
            ln_f = n_seen / moves
        elif flatness >= flat or since_step >= max_rounds_per_step:
            # The second condition is a guarantee, not an optimisation. With many bins the
            # histogram may never reach the flatness target in any reasonable time, and without
            # this the schedule stalls with ln f at its starting value and the weights run away.
            # Stage one only has to find the SHAPE; `refine` is what makes it accurate.
            ln_f /= 2.0
            hist[:] = 0
            since_step = 0
        rounds.append(dict(round=len(rounds), moves=moves, ln_f=ln_f, flatness=flatness,
                           bins=n_seen, one_over_t=tail))
    return dict(lng=lng, seen=seen, rounds=rounds, moves=moves, ln_f=ln_f,
                accept=accepted / max(moves, 1), discovery_round=discovery_round,
                stalled=ln_f > ln_f_final and moves >= max_moves)


def refine(advance, lng, seen, n_sweeps, seed=0, passes=2, sweeps_per_block=None):
    """Stage two: freeze the weights, count visits, correct. Unbiased, and it converges.

    With ln f set to zero the walk no longer changes its own weights, so it is an ordinary
    Monte Carlo chain sampling bin i with probability proportional to g_i exp(-lng_i). Counting
    visits therefore measures exactly the factor by which lng is wrong, and

        lng  <-  lng + ln(visits) - mean of that

    removes it. Nothing is tuned and there is no schedule to get wrong. Two passes are enough
    once the shape is roughly right; the second one is what shows whether the first converged.

    n_sweeps : sweeps per pass. Split into blocks so that the disagreement between blocks can
               be reported as an error bar (`sweeps_per_block`, default one tenth of a pass).
    Returns a dict: lng, seen, flatness (the worst visit deficit of the last pass, which should
    approach 1), err (per-bin standard error in ln g from the blocks), passes (one row each),
    and `unvisited`, the bins the frozen walk never reached. A bin in `unvisited` keeps whatever
    stage one gave it and must be treated as unmeasured.
    """
    lng = lng.copy()
    block = sweeps_per_block or max(1, n_sweeps // 10)
    first = -1 if seed is None else int(np.random.SeedSequence(seed).generate_state(1)[0])
    rows, err, started = [], np.zeros_like(lng), False
    for p in range(passes):
        blocks = []
        for _ in range(max(1, n_sweeps // block)):
            hist = np.zeros(lng.shape, dtype=np.int64)
            advance(lng, hist, seen, 0.0, block, -1 if started else first)   # seed once (Q14)
            started = True
            blocks.append(hist)
        total = np.sum(blocks, axis=0)
        hit = total > 0
        correction = np.where(hit, np.log(np.maximum(total, 1)), 0.0)
        correction -= correction[hit].mean()
        lng = np.where(hit, lng + correction, lng)
        share = np.array([b / max(b.sum(), 1) for b in blocks])
        with np.errstate(invalid="ignore", divide="ignore"):
            err = np.where(hit, share.std(axis=0, ddof=1) / np.maximum(share.mean(axis=0), 1e-300)
                           / np.sqrt(len(blocks)), np.nan)
        counts = total[hit]
        rows.append(dict(pass_=p, sweeps=n_sweeps, blocks=len(blocks), bins=int(hit.sum()),
                         flatness=float(counts.min() / counts.mean()),
                         worst_err=float(np.nanmax(err))))
    return dict(lng=lng, seen=seen & hit, err=err, passes=rows,
                flatness=rows[-1]["flatness"], unvisited=seen & ~hit)


def normalise_to(lng, seen, ref, value):
    """Shift ln g so that bin `ref` holds `value` (a count, not a log). Returns a fresh array."""
    out = np.where(seen, lng - lng[ref] + np.log(value), np.nan)
    return out


def spread(runs, seen):
    """Largest disagreement in ln g between independent runs, after shifting each to a common bin.

    `runs` are the lng arrays; `seen` the bins to compare on. Returns (max difference, per-bin
    standard deviation). Both are in units of ln g, so 0.1 means 10 % in g.
    """
    where = np.argwhere(seen)
    ref = tuple(where[0])
    shifted = np.array([np.where(seen, r - r[ref], np.nan) for r in runs])
    with warnings.catch_warnings():                 # bins outside `seen` are all-NaN on purpose
        warnings.simplefilter("ignore", RuntimeWarning)
        lo, hi = np.nanmin(shifted, axis=0), np.nanmax(shifted, axis=0)
        return float(np.nanmax(hi - lo)), np.nanstd(shifted, axis=0)


def canonical(lng, seen, s_min, x_min, n, g, lam):
    """Averages at coupling g and knob lam from the density of states.

    Returns (phi, surplus per vertex, energy per vertex, heat capacity per vertex). The sum is
    done with the largest term factored out, so it does not overflow at small g.
    """
    ss, xs = np.nonzero(seen)
    s = ss + s_min
    x = xs + x_min
    h = ENERGY_PER_SQUARE * (n - s) + ENERGY_PER_SURPLUS * lam * x
    w = lng[seen] - h / g
    w -= w.max()
    p = np.exp(w)
    p /= p.sum()
    e = float((p * h).sum())
    e2 = float((p * h * h).sum())
    return float((p * s).sum()) / n, float((p * x).sum()) / n, e / n, (e2 - e * e) / (g * g * n)


def energy_histogram(lng, seen, s_min, x_min, n, g, lam, places=6):
    """The curve VISION S2 is about: P(H) at coupling g, as (energies, probabilities).

    Bins of equal H are added together (`places` rounds away the floating-point noise in
    lambda H). The probabilities are normalised to sum to 1.
    """
    ss, xs = np.nonzero(seen)
    h = ENERGY_PER_SQUARE * (n - (ss + s_min)) + ENERGY_PER_SURPLUS * lam * (xs + x_min)
    w = lng[seen] - h / g
    w -= w.max()
    p = np.exp(w)
    levels = {}
    for value, weight in zip(np.round(h, places), p):
        levels[value] = levels.get(value, 0.0) + weight
    out = np.array(sorted(levels.items()))
    out[:, 1] /= out[:, 1].sum()
    return out[:, 0], out[:, 1]


@njit(cache=True)
def _wl_sweeps(adj, side_u, lng, hist, seen, ln_f, n_sweeps, seed, s_min, x_min, cap, state, floor,
               track_x):
    """Wang-Landau sweeps of the graph model. state = [S, X], read and written in place.

    track_x = False walks in S alone: X is neither computed nor binned (the grid is n_s by 1 and
    j is always 0). That is exact at lambda = 0, where X does not enter the energy, and it turns
    a walk over hundreds of (S, X) bins into one over tens of S bins.

    The move, the validity rules and the bookkeeping of Delta X are the kernel's own
    (cqg.run_chain); only the acceptance rule differs. A move out of the grid is refused,
    which restricts the walk to the window and leaves it correct inside it. A bin not yet
    reached counts as holding `floor`, the smallest ln g among the bins seen so far, rather
    than zero: a fresh bin carries no information, and treating it as zero would make the
    walk pour into it. Returns (attempted moves, accepted moves).

    seed < 0 carries on the random stream from the previous call instead of starting a new
    one, and that is what every call after the first must do. Re-seeding between short blocks
    visibly spoils the walk: measured on the Ising model with the weights held at the exact
    answer and the total work fixed, the flat histogram went from 1.5 % uneven in one call to
    12 % uneven in a thousand re-seeded blocks (ASSUMPTIONS Q14).
    """
    if seed >= 0:
        np.random.seed(seed)
    n = adj.shape[0]
    nu = side_u.shape[0]
    n_s, n_x = lng.shape
    edges = np.empty((64, 2), dtype=np.int64)
    s, x = state[0], state[1]
    # state[2] = which end of the window was touched last (0 neither, 1 low, 2 high)
    # state[3] = completed round trips, low end to high end and back (ASSUMPTION Q17)
    last_end, trips = state[2], state[3]
    low_s, high_s = 0, n_s - 1
    attempted, accepted = 0, 0
    for _ in range(n_sweeps):
        for _ in range(2 * n):
            attempted += 1
            u1 = side_u[np.random.randint(0, nu)]
            v1 = adj[u1, np.random.randint(0, 4)]
            u2 = side_u[np.random.randint(0, nu)]
            v2 = adj[u2, np.random.randint(0, 4)]
            if not (u1 == u2 or v1 == v2 or has_edge(adj, u1, v2) or has_edge(adj, u2, v1)):
                n_edges = 0
                if track_x:
                    n_edges = _note_square_edges(adj, u1, v1, edges, n_edges)
                    n_edges = _note_square_edges(adj, u2, v2, edges, n_edges)
                lost = squares_on_edge(adj, u1, v1)
                _replace(adj, u1, v1, -1)
                _replace(adj, v1, u1, -1)
                lost += squares_on_edge(adj, u2, v2)
                _replace(adj, u2, v2, -1)
                _replace(adj, v2, u2, -1)
                _replace(adj, u1, -1, v2)
                _replace(adj, v2, -1, u1)
                gained = squares_on_edge(adj, u1, v2)
                _replace(adj, u2, -1, v1)
                _replace(adj, v1, -1, u2)
                gained += squares_on_edge(adj, u2, v1)
                d_s = gained - lost
                d_x = 0
                ok = _new_edge_ok(adj, u1, v2, cap) and _new_edge_ok(adj, u2, v1, cap)
                if ok and track_x:
                    n_edges = _note_square_edges(adj, u1, v2, edges, n_edges)
                    n_edges = _note_square_edges(adj, u2, v1, edges, n_edges)
                    after = _surplus_on(adj, edges, n_edges)
                    _switch(adj, u1, v2, u2, v1)
                    before = _surplus_on(adj, edges, n_edges)
                    _switch(adj, u1, v1, u2, v2)
                    d_x = after - before
                i, j = s + d_s - s_min, x + d_x - x_min
                if ok and (i < 0 or i >= n_s or j < 0 or j >= n_x):
                    ok = False                       # outside the window
                if ok:
                    there = lng[i, j] if seen[i, j] else floor
                    d_l = lng[s - s_min, x - x_min] - there
                    if d_l < 0.0 and np.random.random() >= np.exp(d_l):
                        ok = False
                if ok:
                    s += d_s
                    x += d_x
                    accepted += 1
                else:
                    _switch(adj, u1, v2, u2, v1)
            i, j = s - s_min, x - x_min
            if not seen[i, j]:
                seen[i, j] = True
                lng[i, j] = floor            # a fresh bin starts level with the lowest
            lng[i, j] += ln_f
            hist[i, j] += 1
            if i == low_s:
                if last_end == 2:
                    trips += 1               # came back from the far end: one round trip
                last_end = 1
            elif i == high_s:
                last_end = 2
    state[0], state[1] = s, x
    state[2], state[3] = last_end, trips
    return attempted, accepted


def graph_sweeper(adj, side_u, s_min, x_min, cap=NO_CAP, track_x=None):
    """An `advance` function for `anneal` that walks the graph model. Modifies adj in place.

    `advance.state` is [S, X, which end was last touched, completed round trips]. The last of
    these is the diagnostic the T6 pre-registration requires with every run: a flat-histogram
    walk that has not crossed its window many times has not measured the far end (Q17).

    track_x defaults to "whenever the cap is off", which is when X can vary. Pass False to walk
    in S alone (one-dimensional; exact at lambda = 0), in which case x_min is ignored.
    """
    if track_x is None:
        track_x = cap > CAP
    if not track_x:
        x_min = 0
    state = np.array([total_squares(adj), surplus(adj) if track_x else 0, 0, 0], dtype=np.int64)

    def advance(lng, hist, seen, ln_f, n_sweeps, seed):
        floor = float(lng[seen].min()) if seen.any() else 0.0
        return _wl_sweeps(adj, side_u, lng, hist, seen, ln_f, n_sweeps, seed,
                          s_min, x_min, cap, state, floor, track_x)

    advance.state = state
    return advance


def both_stages(advance, shape, seed=0, refine_sweeps=0, passes=2, **kw):
    """`anneal` then `refine`, which is how this should always be run (see the module docstring).

    refine_sweeps = 0 skips stage two and returns stage one alone, which is for looking at the
    schedule, not for reporting a result.
    """
    out = anneal(advance, shape, seed=seed, **kw)
    out["stage_one_lng"] = out["lng"].copy()
    out["round_trips_stage_one"] = int(getattr(advance, "state", [0, 0, 0, 0])[3])
    if refine_sweeps:
        second = refine(advance, out["lng"], out["seen"], refine_sweeps, seed=None, passes=passes)
        out.update(lng=second["lng"], seen=second["seen"], err=second["err"],
                   passes=second["passes"], flatness=second["flatness"],
                   unvisited=second["unvisited"])
    out["round_trips"] = int(getattr(advance, "state", [0, 0, 0, 0])[3])
    return out


def dos_graph(adj, side_u, s_range, x_range=None, cap=NO_CAP, **kw):
    """Density of states of the graph model over a window of (S, X). Modifies adj in place.

    s_range, x_range : inclusive (low, high) pairs. The walk is refused outside them, which
    is a legitimate restriction: ln g comes out right inside the window, up to the usual
    constant, provided the window is connected under the move.

    x_range = None walks in S alone and returns ln g with shape (n_s, 1). This is exact at
    lambda = 0, where X does not enter the energy, and it is the difference between a walk
    over hundreds of bins and one over tens.

    Returns the dict of `both_stages` with s_min and x_min added.
    """
    s_min, s_max = s_range
    if x_range is None:
        advance = graph_sweeper(adj, side_u, s_min, 0, cap, track_x=False)
        out = both_stages(advance, (s_max - s_min + 1, 1), **kw)
        out["s_min"], out["x_min"] = s_min, 0
        return out
    x_min, x_max = x_range
    advance = graph_sweeper(adj, side_u, s_min, x_min, cap)
    out = both_stages(advance, (s_max - s_min + 1, x_max - x_min + 1), **kw)
    out["s_min"], out["x_min"] = s_min, x_min
    return out
