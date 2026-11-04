"""T71's layer-form protocol with surroundings and a curl edge (PREREGISTRATION.md T74 and its Amendments 2 and 3;
VISION Updates 72 and 73; ASSUMPTIONS Q24; the curling-energy brief, section "Proposed: energy that travels to the curl
edge").

Naming: at lambda != 1, and with a varying count of points, this is our family around the published model, never
combinatorial quantum gravity (VISION Update 22).

The slab and its moves are T71's (`graphity.layer_feed`, reused here, not copied): the layer moves of `contraction.py`
(`propose_layer_move`), the clean two-switch move (`clean_move.run_block`), the gate, and a release that leaves into a
daughter ledger. What T74 adds is where the feed goes. Nothing feeds the slab's stores directly:

* **The surroundings.** One reservoir of `n_out` stores holding E_out (g_out = E_out / n_out). The feed is added to
  E_out. In a sweep where the reservoir relaxes, a fraction rho of (E_out - n_out g_bg) moves to the far-space ledger
  `far` (or, when negative, from it; `far` can go below zero: it is a ledger, not a store).
* **The curl edge.** Then energy moves between the reservoir and the slab's stores, downhill in mean energy per store
  (g_out against g_in = sum(stores) / N, the mean over the slab's points on its current count), by the smaller of the
  cap `c` and the amount x* that would make the two means equal, x* = (E_out N - S_in n_out) / (N + n_out). One cap,
  both directions (the author's (a), Update 73). Inflow is spread evenly over the slab's stores. Outflow is taken
  evenly from them, except that no store is taken below zero: a store holding less than the even share gives all it
  has and the rest is shared evenly among the others (water-filling; ours, ASSUMPTIONS Q24); when the stores are equal
  it is the plain even take. The amount moved is always exactly the flow, since x* < S_in.
* **Three phases** (Amendments 2 and 3):
  1. **fed**, sweeps 1..n_fed: the feed into E_out, "steady" (`amount` every sweep) or "meals" (`amount` every `every`
     sweeps, the first at sweep 1: a meal arrives at the start of its period, ours, Q24); the reservoir relaxes at
     `rho_fed` toward `g_bg_fed` (rho_fed = 0 under Amendment 2: isolated, the meals its only input);
  2. **digestion**, from sweep n_fed + 1 (Amendment 3, item 1): no feed, the reservoir still relaxing at `rho_fed`
     (isolated), until the reservoir's mean store energy and the slab's agree within `digest_tol` (1 %),
     |g_out - g_in| <= digest_tol g_in, read at the end of a sweep after its moves, or `n_digest_max` sweeps have
     passed, whichever is first. The check is made from the end of sweep n_fed on, so a fed phase that ends balanced
     has no digestion. Its last sweep, t_unfed, is "the moment" of Amendment 3;
  3. **unfed**, sweeps t_unfed + 1 .. t_unfed + n_unfed: no feed, the reservoir relaxing at `rho_unfed` (1/64) toward
     `g_bg_unfed`: cold (0), warm (4a), or EQUAL, the common mean of reservoir and slab stores at t_unfed,
     (E_out + sum(stores)) / (n_out + N) (Amendment 3, item 2).
  With an Outside the run's length is set by these phases, not by the caller.
* **Initial state of the surroundings** (ours, Q24): E_out = `e_out0` (default 0) and far = 0.

One sweep, in this order (the registration: "after the feed and the relaxation and before the moves"):
  1. the feed into E_out (fed phase, per schedule);
  2. the relaxation of E_out toward n_out g_bg;
  3. the curl-edge flow;
  4. the leak of a fraction `leak` of every slab store (T71's; 0 in T74), counted in `leaked`;
  5. N clean two-switch attempts (`move_set` "model"; none for "layer"; clean check off for "breaking");
  6. `proposals_per_sweep` layer proposals, exactly as in `run_layer_feed`.
With `outside=None`, steps 1 to 3 are replaced by T71's feed (`rate` into every slab store, stopping at the first split
if `feed_stops_at_split`), the run is `n_sweeps` long, and the function is `run_layer_feed` draw for draw, rows and all
(a test checks it).

Energy is conserved exactly:
    F + sum(stores) + daughter_energy + E_out + far - fed + leaked = F0 + e_out0
with F = H + mu_N N (tests, to 1e-9; with e_out0 = 0 this is the registered invariant). The protocol is not an
equilibrium chain and claims no detailed balance; conservation is its invariant, as for T71.
"""
from dataclasses import dataclass

import numpy as np

from . import clean_move as CM
from .contraction import energy
from .layer_feed import MOVE_SETS, RELEASE_TO, census, propose_layer_move

FEEDS = ("steady", "meals")
EQUAL = "equal"
FED, DIGESTION, UNFED = 1, 2, 3


@dataclass(frozen=True)
class Outside:
    """The surroundings and the curl edge of T74. Energies per sweep; `amount` is what one feeding adds to E_out."""
    n_out: int
    c: float
    n_fed: int
    feed: str = "steady"
    amount: float = 0.0
    every: int = 1
    e_out0: float = 0.0
    rho_fed: float = 0.0
    g_bg_fed: float = 0.0
    rho_unfed: float = 1.0 / 64
    g_bg_unfed: object = 0.0          # a number, or EQUAL: the common mean of reservoir and slab at the unfed start
    n_digest_max: int = 4096
    digest_tol: float = 0.01
    n_unfed: int = 4096

    def __post_init__(self):
        assert self.feed in FEEDS and self.n_out > 0 and self.c >= 0 and self.every >= 1 and self.n_fed >= 1
        assert 0 <= self.rho_fed <= 1 and 0 <= self.rho_unfed <= 1
        assert self.n_digest_max >= 0 and self.digest_tol >= 0 and self.n_unfed >= 0
        assert self.g_bg_unfed == EQUAL or isinstance(self.g_bg_unfed, (int, float))
        if self.feed == "steady":
            assert self.every == 1

    def fed_at(self, sweep):
        """What the feed adds to E_out in this sweep (1-based)."""
        if sweep > self.n_fed:
            return 0.0
        if self.feed == "steady":
            return self.amount
        return self.amount if (sweep - 1) % self.every == 0 else 0.0

    def relaxation(self, phase, g_common=None):
        """(rho, g_bg) in a sweep of this phase; `g_common` is the common mean at the unfed start (for EQUAL)."""
        if phase != UNFED:
            return self.rho_fed, self.g_bg_fed
        return self.rho_unfed, (g_common if self.g_bg_unfed == EQUAL else float(self.g_bg_unfed))

    def balanced(self, e_out, stores):
        """The end of digestion: the two means agree within digest_tol of the slab's."""
        g_out, g_in = e_out / self.n_out, float(stores.sum()) / stores.shape[0]
        return abs(g_out - g_in) <= self.digest_tol * g_in


def take_evenly(stores, x):
    """Remove x >= 0 from `stores` in place, evenly, no store below zero (water-filling). Requires x <= sum(stores)."""
    if x <= 0.0:
        return
    order = np.argsort(stores, kind="stable")
    left = x
    n_left = stores.shape[0]
    for idx in order:
        share = left / n_left
        give = min(share, stores[idx])
        stores[idx] -= give
        left -= give
        n_left -= 1
        if n_left == 0:
            break
    assert left <= 1e-9 * max(1.0, x), "took more than the stores hold"


def edge_flow(e_out, n_out, stores, c):
    """The curl-edge flow into the slab (negative: out of it) for this sweep. Returns (flow, g_out, g_in) with the
    means measured before the flow."""
    n_in = stores.shape[0]
    s_in = float(stores.sum())
    g_out, g_in = e_out / n_out, s_in / n_in
    x_eq = (e_out * n_in - s_in * n_out) / (n_in + n_out)
    flow = float(np.sign(x_eq) * min(c, abs(x_eq)))
    return flow, g_out, g_in


def run_layer_outside(state, lam, n_sweeps, seed, outside=None, rate=0.0, gate=0.0, leak=0.0,
                      feed_stops_at_split=False, move_set="model", max_width=8, axes=(0,), proposals_per_sweep=1,
                      mu_n=0.0, log_every=1, release_to="stores", log_events=True):
    """The protocol of the module docstring. Returns (final_state, rows, summary).

    n_sweeps: the run's length with outside=None; with an Outside it must be None (the phases set the length).
    rows: with outside=None, exactly `run_layer_feed`'s 15-tuples. With an Outside, each row is that tuple followed by
    (E_out, far, edge_in, flow, g_out_before, g_in_before, n_in_before, g_bg, phase): edge_in is the cumulative net flow
    into the slab, flow this sweep's flow, g_out_before and g_in_before the means just before it, n_in_before the
    slab's count of points then (the moves after it can change the count), g_bg the background in force that sweep,
    phase 1 (fed), 2 (digestion) or 3 (unfed). The sweeps n_fed and t_unfed are always logged.
    summary: first_split, first_clean, first_contract (sweeps or None), f0, and with an Outside: e_out0, max_abs_flow,
    births (a list of (sweep, energy) for every daughter), t_unfed (the last sweep of digestion), digest_balanced
    (whether digestion ended by balance rather than by n_digest_max), g_common (the common mean of reservoir and slab
    stores at the end of t_unfed) and s_unfed_start (the slab's stored energy, the sum of its stores, then).
    """
    assert move_set in MOVE_SETS and release_to in RELEASE_TO
    o = outside
    assert (o is None) == (n_sweeps is not None), "n_sweeps is the length only without an Outside"
    rng = np.random.default_rng(seed)
    np.random.seed(int(rng.integers(2**31 - 1)))
    CM.seed_numba(int(rng.integers(2**31 - 1)))
    st = state.copy()
    f = energy(st, lam, mu_n)
    f0 = f
    fed = leaked = 0.0
    n_split = n_contract = n_clean = 0
    daughters, daughter_energy = 0, 0.0
    first = dict(first_split=None, first_clean=None, first_contract=None)
    work = CM.work_arrays(st.adj.shape[1])
    side0 = np.flatnonzero(st.part == 0).astype(np.int64)
    rows = []
    if o is not None:
        e_out, far, edge_in = float(o.e_out0), 0.0, 0.0
        max_abs_flow = 0.0
        births = []
        phase, t_unfed, g_common, s_unfed_start, digest_balanced = FED, None, None, None, None
        end = None                                   # the last sweep, known once digestion ends
    sweep = 0
    while True:
        sweep += 1
        if o is None and sweep > n_sweeps:
            break
        if o is not None and end is not None and sweep > end:
            break
        before = (n_split, n_contract, n_clean, daughters)
        if o is None:
            if not (feed_stops_at_split and first["first_split"] is not None):
                st.stores += rate
                fed += rate * st.n
        else:
            meal = o.fed_at(sweep)
            e_out += meal
            fed += meal
            rho, g_bg = o.relaxation(phase, g_common)
            to_far = rho * (e_out - o.n_out * g_bg) if rho > 0 else 0.0
            e_out -= to_far
            far += to_far
            n_in_b = st.n
            flow, g_out_b, g_in_b = edge_flow(e_out, o.n_out, st.stores, o.c)
            if flow > 0.0:
                st.stores += flow / st.n
            elif flow < 0.0:
                take_evenly(st.stores, -flow)
            e_out -= flow
            edge_in += flow
            max_abs_flow = max(max_abs_flow, abs(flow))
        if leak > 0:
            out = leak * st.stores
            st.stores -= out
            leaked += float(out.sum())
        if move_set != "layer":
            acc, dh_sum = CM.run_block(st.adj, st.part, st.stores, side0, lam, move_set == "breaking", 1, 0.0, *work)
            if acc:
                n_clean += acc
                f += dh_sum
                if first["first_clean"] is None:
                    first["first_clean"] = sweep
        for _ in range(proposals_per_sweep):
            st, f, kind, released = propose_layer_move(st, f, lam, mu_n, rng, axes, max_width, gate, release_to)
            if kind is None:
                continue
            if release_to == "daughter" and released > 1e-9:
                daughters += 1
                daughter_energy += released
                if o is not None:
                    births.append((sweep, released))
            side0 = np.flatnonzero(st.part == 0).astype(np.int64)
            if kind == "split":
                n_split += 1
                if first["first_split"] is None:
                    first["first_split"] = sweep
            else:
                n_contract += 1
                if first["first_contract"] is None:
                    first["first_contract"] = sweep
        fired = (n_split, n_contract, n_clean, daughters) != before
        boundary = False
        if o is not None:
            row_phase = phase
            if sweep == o.n_fed:
                boundary = True
                phase = DIGESTION
            if phase == DIGESTION:
                ok = o.balanced(e_out, st.stores)
                if ok or sweep - o.n_fed >= o.n_digest_max:
                    boundary = True
                    phase, t_unfed, digest_balanced = UNFED, sweep, bool(ok)
                    s_unfed_start = float(st.stores.sum())
                    g_common = (e_out + s_unfed_start) / (o.n_out + st.n)
                    end = t_unfed + o.n_unfed
        last = (o is None and sweep == n_sweeps) or (o is not None and end is not None and sweep == end)
        if sweep % log_every == 0 or last or (log_events and fired) or boundary:
            op, cu, br = census(st.adj)
            row = (sweep, st.n, tuple(st.dims), float(f), float(st.stores.sum()), fed, leaked, n_split,
                   n_contract, n_clean, op, cu, br, daughters, daughter_energy)
            if o is not None:
                row = row + (e_out, far, edge_in, flow, g_out_b, g_in_b, n_in_b, g_bg, row_phase)
            rows.append(row)
    first["f0"] = f0
    if o is not None:
        first.update(e_out0=float(o.e_out0), max_abs_flow=max_abs_flow, births=births, t_unfed=t_unfed,
                     digest_balanced=digest_balanced, g_common=g_common, s_unfed_start=s_unfed_start)
    return st, rows, first


def conservation_residuals(f0, rows, e_out0=0.0):
    """F + stores + daughter + E_out + far - fed + leaked - (F0 + e_out0) for every logged row of the outside path."""
    return [r[3] + r[4] + r[14] + r[15] + r[16] - r[5] + r[6] - f0 - e_out0 for r in rows]
