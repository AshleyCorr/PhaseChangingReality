"""Parallel tempering (replica exchange) for the 2D CQG model (task T5).

THE PROBLEM. At low coupling g a single chain freezes: almost every switch is
refused, so it stays wherever it happened to be (acceptance below 1 %; ASSUMPTIONS,
first-look section). Its averages then describe its history, not equilibrium.

THE METHOD [NB99, from general knowledge; the standard technique]. Run one copy of
the system at each of several couplings g_1 > g_2 > ... > g_K, all at once. Each
copy does ordinary sweeps (cqg.run_chain). Every so often two copies at
neighbouring couplings are offered the chance to swap their whole graphs. A hot
copy moves freely; through swaps, what it finds can travel down to the cold
couplings, and a stuck cold graph can travel up, get shaken loose and come back.

WHY IT IS STILL EXACT. Think of all K copies together as one big system whose
weight is the product of the K usual weights exp(-H_k / g_k). Swapping the graphs
at couplings i and j changes that product by the factor
    a = exp( (1/g_i - 1/g_j) * (H_i - H_j) ),
and the swap is accepted with probability min(1, a) (Metropolis). The proposal is
symmetric, so this is detailed balance for the big system, and the sweeps in
between are detailed balance for each copy. So every coupling still samples
exp(-H / g), exactly as a single chain would if it could move. Tested against
the exact averages at N = 18 (tests/test_tempering.py).

ASSUMPTION Q11 (ours: the details). Swaps are offered between neighbours only,
pairs (1,2), (3,4), ... on even rounds and (2,3), (4,5), ... on odd rounds. One
round = `sweeps_per_round` sweeps of every copy, then one set of swap offers.

Random numbers, CHANGED 2026-09-20 (ASSUMPTION Q14; this changes results, and the
runs made before it are listed there). The sweeps used to take a fresh seed for
every (round, copy). That is wrong when a round is short: re-seeding the generator
thousands of times measurably spoils the chain, and these runs re-seed it about two
hundred thousand times, five sweeps apart. The sweeps now take one seed at the start
and carry the same stream on through every round and copy. The swap decisions keep
their own separate stream. A run is still reproducible from a single seed.
"""
import numpy as np

from graphity.cqg import ENERGY_PER_SQUARE, ENERGY_PER_SURPLUS, run_chain


def temper(graphs, side_u, couplings, n_rounds, sweeps_per_round, seed_sequence, lam, cap, glauber=False,
           measure_from=0, on_round=None):
    """Run parallel tempering. Modifies `graphs` (one adjacency array per coupling) in place.

    couplings     : g values, hottest first. graphs[k] starts at couplings[k].
    measure_from  : first round whose sweeps are recorded (earlier rounds are equilibration).
    Returns a dict: per coupling the series of S, X and the connectivity numbers over all recorded
    sweeps, the acceptance rate of ordinary switches, the rate at which swaps with the next colder
    coupling were accepted, `round_trips`, how many times a graph went from the hottest coupling
    to the coldest and back (the usual sign that tempering is doing its job), and `at_cold_end`,
    which of the starting graphs (numbered by the coupling it started at) sat at the coldest coupling
    after each round.

    on_round      : optional function on_round(r, graphs), called after the swaps of every recorded
                    round, with graphs[k] the graph now at couplings[k]. It must only read the graphs.
                    It draws no random numbers, so a run with it is bit for bit the run without it
                    (tests/test_tempering.py). Added for T16's snapshots.
    """
    k_total = len(couplings)
    if len(graphs) != k_total or any(a <= b for a, b in zip(couplings, couplings[1:])):
        raise ValueError("one graph per coupling, and couplings strictly decreasing (hottest first)")
    n = graphs[0].shape[0]
    inv_g = [1.0 / g for g in couplings]
    sweep_seeds, swap_seed = seed_sequence.spawn(2)
    rng = np.random.default_rng(swap_seed)
    sweep_seed = int(sweep_seeds.generate_state(1)[0])      # once; then -1 carries the stream on (Q14)
    recorded = (n_rounds - measure_from) * sweeps_per_round
    s_out = np.zeros((k_total, recorded), dtype=np.int64)
    x_out = np.zeros((k_total, recorded), dtype=np.int64)
    conn_out = np.zeros((k_total, recorded, 4), dtype=np.int64)
    accepted = np.zeros(k_total)
    swaps_offered = np.zeros(k_total - 1)
    swaps_taken = np.zeros(k_total - 1)
    # which starting graph sits at each coupling; stage 1 = has been at the hot end, 2 = and then at the cold end
    walker = list(range(k_total))
    stage = [0] * k_total
    stage[0] = 1
    round_trips = 0
    at_cold_end = np.zeros(n_rounds, dtype=np.int64)
    for r in range(n_rounds):
        energy = np.zeros(k_total)
        for k in range(k_total):
            conn = np.zeros((sweeps_per_round, 4), dtype=np.int64)
            s, x, acc = run_chain(graphs[k], side_u, inv_g[k], 0, sweeps_per_round,
                                  sweep_seed if (r == 0 and k == 0) else -1, lam, cap, glauber, conn)
            energy[k] = ENERGY_PER_SQUARE * (n - s[-1]) + ENERGY_PER_SURPLUS * lam * x[-1]
            accepted[k] += acc
            if r >= measure_from:
                at = (r - measure_from) * sweeps_per_round
                s_out[k, at:at + sweeps_per_round] = s
                x_out[k, at:at + sweeps_per_round] = x
                conn_out[k, at:at + sweeps_per_round] = conn
        for k in range(r % 2, k_total - 1, 2):
            swaps_offered[k] += 1
            log_a = (inv_g[k] - inv_g[k + 1]) * (energy[k] - energy[k + 1])
            if log_a >= 0 or rng.random() < np.exp(log_a):
                swaps_taken[k] += 1
                graphs[k], graphs[k + 1] = graphs[k + 1], graphs[k]
                energy[k], energy[k + 1] = energy[k + 1], energy[k]
                walker[k], walker[k + 1] = walker[k + 1], walker[k]
        hot, cold = walker[0], walker[-1]
        if k_total > 1 and stage[cold] == 1:
            stage[cold] = 2                            # hot end, then cold end: half a trip
        if k_total > 1 and stage[hot] == 2:
            round_trips += 1                           # and back at the hot end: a full trip
        stage[hot] = 1
        at_cold_end[r] = cold
        if on_round is not None and r >= measure_from:
            on_round(r, graphs)
    return dict(squares=s_out, surplus=x_out, connectivity=conn_out, acceptance=accepted / n_rounds,
                swap_rate=swaps_taken / np.maximum(swaps_offered, 1), round_trips=round_trips,
                at_cold_end=at_cold_end)
