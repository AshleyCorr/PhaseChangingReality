"""T76's settling runs alone, for the bisection that fixes its couplings. Usage:

    python scripts/settle_t76.py configs/t76_settle_<name>.json [out_dir]

PREREGISTRATION T76 fixes its four couplings by a rule: those at which the settled flat 16 x 16 sheet has
rho_eq = 0.003, 0.01, 0.03 and 0.1, found by bisection on settling runs alone (growth off) and written into an
amendment before any growth runs. This script measures rho_eq at the couplings a config lists; the bisection itself
(which couplings to try next, and when to stop) is the amendment's, and each round is its own config, so every
coupling tried is kept (rule 5; S1).

Two starts. Without "states" in the config (rounds 1 to 4), each replica is exactly the settling stage of
scripts/run_t76.py: the flat L x L torus, lambda, no cap, graphity.row_growth.run_growth with growth off for
`n_settle` fair sweeps. With "states_dir" and "states" (T76 Amendment 4, the owner's option 1 of 20 October: the
burp's product), replica r starts from the r-th saved end state of that T51 run (scripts/run_t75.load_states, in
order of replica number, reused cyclically when there are fewer states than replicas, each reuse recorded) and runs
graphity.row_growth.settle_graph, the same chain, fair clock and stream with no grid labels. A product already holds
knots, whose missing squares count in D; so each row also gives rho_start, the density before any attempt, and
excess = rho_eq - rho_start, the part the coupling added.

rho_eq is the mean of rho = D / N over the readings of the last `window` fair sweeps (analyse_t76.rho_eq). Streams
from SeedSequence([seed, k, replica]) for the k-th coupling of the list.

Rows: one per (coupling, replica): g, N, replica, state (the saved file, or "torus"), reuse (how many earlier
replicas of this coupling started from the same state), rho_start, rho_eq, excess, rho_sd (spread of rho over the
window), D_mean, curled_share (mean over the window) and the window's mean acceptance. Prints, per coupling, the
mean rho_eq and excess over replicas, the median excess and the number of replicas whose rho_eq passes 1 (melted).

Config keys: name, section ("T76-settle"), lambda, couplings (a list), replicas, n_settle, window, n_ref, seed, and
either L (the flat torus) or states_dir and states (the product). At lambda != 1 this is our family around the
published model, not combinatorial quantum gravity.
"""
import json
import platform
import sys
from pathlib import Path

import numba
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from graphity import __version__                                                  # noqa: E402
from graphity.results import ResultWriter, provenance                             # noqa: E402
from graphity.row_growth import GrowthState, run_growth, settle_graph             # noqa: E402
from analyse_t76 import rho_eq                                                    # noqa: E402

KEYS = ("name", "section", "lambda", "couplings", "replicas", "n_settle", "window", "n_ref", "seed")
MELTED = 1.0          # rho_eq above this: the replica melted (the random phase sits near 1.3 to 1.7, O141)


def summarize(recs, n_settle, window):
    """The window's readings of one settling run, as a dict."""
    win = [r for r in recs if r["phase"] == "settle" and r["fair_sweep"] > n_settle - window]
    eq = rho_eq(recs, n_settle, window)
    return dict(rho_start=float(recs[0]["rho"]), rho_eq=eq, excess=eq - float(recs[0]["rho"]),
                rho_sd=float(np.std([r["rho"] for r in win])), D_mean=float(np.mean([r["D"] for r in win])),
                curled_share=float(np.mean([r["curled_share"] for r in win])),
                acceptance=float(np.mean([r["acceptance"] for r in win])))


def settle(size, g, lam, n_settle, window, n_ref, seed):
    """One replica's settling run on the flat torus: a dict of the window's readings."""
    recs = run_growth(GrowthState.from_torus(int(size)), float(g), float(lam), int(n_settle), 0, 8, seed=seed,
                      n_ref=int(n_ref))
    return summarize(recs, int(n_settle), int(window))


def settle_state(adj, part, g, lam, n_settle, window, n_ref, seed):
    """One replica's settling run from a saved state (copied, not changed): a dict of the window's readings."""
    recs = settle_graph(adj.copy(), part, float(g), float(lam), int(n_settle), seed=seed, n_ref=int(n_ref))
    return summarize(recs, int(n_settle), int(window))


def starts(cfg):
    """[(label, adj, part, reuse)] for each replica, or None for the flat torus."""
    if "states" not in cfg:
        return None
    from run_t75 import load_states
    states = load_states(cfg["states_dir"], cfg["states"])
    out = []
    for rep in range(int(cfg["replicas"])):
        path, adj, part, _ = states[rep % len(states)]
        out.append((path.name, adj, part, rep // len(states)))
    return out


def main(path, out_dir="results"):
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    if cfg.get("section") != "T76-settle":
        raise ValueError("not a T76 settling config")
    missing = [k for k in KEYS if k not in cfg]
    if ("L" in cfg) == ("states" in cfg):
        missing.append("exactly one of L or states")
    if "states" in cfg and "states_dir" not in cfg:
        missing.append("states_dir")
    if missing:
        raise ValueError("config lacks %s" % ", ".join(missing))
    if not 0 < int(cfg["window"]) <= int(cfg["n_settle"]):
        raise ValueError("the window must lie inside the settling")
    begin = starts(cfg)
    meta = dict(config=cfg, config_path=str(path), package=__version__, python=platform.python_version(),
                numpy=np.__version__, numba=numba.__version__, **provenance(),
                preregistration="PREREGISTRATION.md section T76 (the couplings' bisection; Amendment 4 for a "
                                "product start)")
    summary = []
    with ResultWriter(cfg["name"], meta, out_dir) as out:
        for k, g in enumerate(cfg["couplings"]):
            rows = []
            for rep in range(int(cfg["replicas"])):
                seed = [int(cfg["seed"]), k, rep]
                if begin is None:
                    n, label, reuse = int(cfg["L"]) ** 2, "torus", 0
                    row = settle(cfg["L"], g, cfg["lambda"], int(cfg["n_settle"]), int(cfg["window"]),
                                 cfg["n_ref"], seed)
                else:
                    label, adj, part, reuse = begin[rep]
                    n = adj.shape[0]
                    row = settle_state(adj, part, g, cfg["lambda"], int(cfg["n_settle"]), int(cfg["window"]),
                                       cfg["n_ref"], seed)
                out.write(dict(g=float(g), N=n, lam=float(cfg["lambda"]), replica=rep, state=label, reuse=reuse,
                               **row))
                rows.append(row)
            eq = np.array([r["rho_eq"] for r in rows])
            ex = np.array([r["excess"] for r in rows])
            se = float(np.std(eq, ddof=1) / np.sqrt(len(eq))) if len(eq) > 1 else float("nan")
            melted = int((eq > MELTED).sum())
            summary.append(dict(g=float(g), rho_eq=float(eq.mean()), se=se, excess=float(ex.mean()),
                                excess_median=float(np.median(ex)), melted=melted))
            print("g = %g: rho_eq = %.5f +- %.5f, excess %.5f (median %.5f), melted %d of %d"
                  % (g, eq.mean(), se, ex.mean(), np.median(ex), melted, len(eq)), flush=True)
        meta["summary"] = summary


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "results")
