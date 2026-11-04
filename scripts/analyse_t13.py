"""The pre-registered T13 observables and verdict. Usage:

    python scripts/analyse_t13.py 196
    python scripts/analyse_t13.py 196 484 676

Implements PREREGISTRATION.md section T13 as amended on 23 September 2026:

  amendment 1  a jump is a move of more than 0.25 in phi between two couplings on a curve less
               than 12 % apart in g, which need not be neighbouring. The per-step value is
               reported beside it so both readings stay visible.
  amendment 2  equilibrium agreement is read within the starts that pass their round-trip gate;
               a start that fails its gate takes no part in the verdict, and its difference from
               a gate-passing start is reported as a diagnostic, not as agreement or disagreement.

  amendment 3  where a protocol has replicas, each replica is a curve and the protocol shows a
               jump when more than half of them do. The replica mean is computed and reported
               beside it, and a size where the two readings disagree is flagged, so a reader can
               apply either. Decided before N = 484 and N = 676 were read.

WHAT IS MEASURED. phi = S/N against the coupling g, for protocol P (single chains, cold descent
then cold ascent) and protocol E (parallel tempering from two starts). The verdicts need two or
more sizes; a single size returns NO VERDICT however clean it is.
"""
import csv
import sys
from collections import defaultdict

JUMP = 0.25          # phi, T13 "Definitions, fixed now"
WINDOW = 0.12        # fractional separation in g over which a jump may be accumulated
HYSTERESIS = 0.15    # phi, ascent above descent
AGREE = 0.03         # phi, between starts at a rung and between replicas in the crossover
CROSSOVER = (0.3, 0.9)
SWAP = (0.15, 0.6)
TRIPS = {196: 5, 484: 3, 676: 3}     # T13 "Gates"


def largest_move(gs, phis, window=WINDOW):
    """Largest change in phi between two couplings less than `window` apart in g.

    `window=None` restricts it to neighbouring couplings, which is the definition as originally
    written. Points must already be ordered in g.
    """
    best = (0.0, None, None)
    for i in range(len(gs)):
        for j in range(i + 1, len(gs)):
            if window is None:
                if j != i + 1:
                    break
            else:
                if abs(gs[i] - gs[j]) / max(gs[i], gs[j]) >= window:
                    break
            d = abs(phis[j] - phis[i])
            if d > best[0]:
                best = (d, gs[i], gs[j])
    return best


def curve(rows, key="phi"):
    """(gs, phis) ordered by descending g, averaging any duplicate couplings."""
    d = defaultdict(list)
    for r in rows:
        d[float(r["g"])].append(float(r[key]))
    gs = sorted(d, reverse=True)
    return gs, [sum(d[g]) / len(d[g]) for g in gs]


def read_p(rows):
    """Protocol P at one size: hysteresis, and the jump on each leg both ways of reading it."""
    reps = sorted({r["replica"] for r in rows})
    out = {"replicas": len(reps), "legs": {}}

    for leg in ("cool", "heat"):
        sel = [r for r in rows if r["leg"] == leg]
        gs, ph = curve(sel)
        win_mean = largest_move(gs, ph)
        step_mean = largest_move(gs, ph, window=None)
        per_rep = []
        for rep in reps:
            g2, p2 = curve([r for r in sel if r["replica"] == rep])
            per_rep.append({"replica": rep,
                            "windowed": largest_move(g2, p2),
                            "per_step": largest_move(g2, p2, window=None)})
        out["legs"][leg] = {
            "mean_windowed": win_mean, "mean_per_step": step_mean,
            "per_replica": per_rep,
            "mean_jump": win_mean[0] > JUMP,
            "replicas_jumping": sum(1 for r in per_rep if r["windowed"][0] > JUMP),
        }

    gs, cool = curve([r for r in rows if r["leg"] == "cool"])
    _, heat = curve([r for r in rows if r["leg"] == "heat"])
    gaps = [(h - c, g) for g, c, h in zip(gs, cool, heat)]
    out["hysteresis"] = max(gaps)
    out["hysteresis_met"] = out["hysteresis"][0] > HYSTERESIS
    return out


def read_e(rows, n):
    """Protocol E at one size and one start: the gate, the spread, and the jump."""
    need = TRIPS.get(n, 3)
    trips = {}
    for r in rows:
        trips[r["replica"]] = int(r["round_trips"])
    swaps = [float(r["swap_rate"]) for r in rows if r.get("swap_rate") not in (None, "", "nan")]

    spread_bad = []
    d = defaultdict(list)
    for r in rows:
        d[float(r["g"])].append(float(r["phi"]))
    for g in sorted(d, reverse=True):
        v = d[g]
        mu = sum(v) / len(v)
        if CROSSOVER[0] < mu < CROSSOVER[1] and max(v) - min(v) > AGREE:
            spread_bad.append((g, max(v) - min(v)))

    gs, ph = curve(rows)
    gate = (all(t >= need for t in trips.values())
            and bool(swaps) and min(swaps) >= SWAP[0] and max(swaps) <= SWAP[1]
            and not spread_bad)
    return {"trips": trips, "trips_needed": need,
            "swaps": (min(swaps), max(swaps)) if swaps else None,
            "spread_exceeds": spread_bad,
            "gate": gate,
            "windowed": largest_move(gs, ph), "per_step": largest_move(gs, ph, window=None),
            "curve": (gs, ph)}


def agreement(starts):
    """Amendment 2: agreement is read among the starts that pass their gate.

    Returns (verdict, largest difference, rung) where verdict is True, False or None. None means
    fewer than two starts passed, so agreement between starts is not evaluable — which is not the
    same as disagreement, and the caller must not read it as one.
    """
    passing = {k: v for k, v in starts.items() if v["gate"]}
    if len(passing) < 2:
        return None, None, None
    names = sorted(passing)
    worst = (-1.0, None)                     # -1 so that a perfect match still records its rung
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            ga, pa = passing[names[i]]["curve"]
            gb, pb = passing[names[j]]["curve"]
            b = dict(zip(gb, pb))
            for g, p in zip(ga, pa):
                if g in b and abs(p - b[g]) > worst[0]:
                    worst = (abs(p - b[g]), g)
    if worst[1] is None:                     # the starts share no rung, so nothing is comparable
        return None, None, None
    return worst[0] <= AGREE, worst[0], worst[1]


def verdict(sizes):
    """T13's three verdicts. `sizes` maps N to {"p": read_p(...), "e": {start: read_e(...)}}."""
    lines = []
    ready = []

    for n in sorted(sizes):
        s = sizes[n]
        p, e = s["p"], s["e"]
        passing = [k for k, v in e.items() if v["gate"]]
        agree, worst, rung = agreement(e)

        e_jump = any(v["windowed"][0] > JUMP for k, v in e.items() if v["gate"])
        mean_j = p["legs"]["heat"]["mean_jump"]
        reps_j = p["legs"]["heat"]["replicas_jumping"]
        ascent_jump = reps_j > p["replicas"] / 2          # amendment 3: the majority decides
        split = (mean_j != ascent_jump)                   # reported, no longer verdict-changing

        lines.append("N = %d" % n)
        lines.append("  P  hysteresis %.3f at g = %.3f  -> %s"
                     % (p["hysteresis"][0], p["hysteresis"][1],
                        "MET" if p["hysteresis_met"] else "not met"))
        lines.append("  P  ascent jump: %d of %d replicas -> %s; replica mean %.3f (%s)%s"
                     % (reps_j, p["replicas"], "JUMP" if ascent_jump else "no jump",
                        p["legs"]["heat"]["mean_windowed"][0], "jump" if mean_j else "no jump",
                        "   <-- the two readings disagree" if split else ""))
        for k in sorted(e):
            v = e[k]
            lines.append("  E  %-5s trips %s (gate %d) -> %s"
                         % (k, [v["trips"][r] for r in sorted(v["trips"])], v["trips_needed"],
                            "passes" if v["gate"] else "NOT CONVERGED, not interpreted"))
        if agree is None:
            lines.append("  E  agreement between starts: not evaluable, %d start(s) passed the gate"
                         % len(passing))
        else:
            lines.append("  E  agreement between starts: %.3f at g = %.3f -> %s"
                         % (worst, rung, "agree" if agree else "DISAGREE"))
        lines.append("  E  jump among gate-passing starts: %s" % ("yes" if e_jump else "no"))

        if not passing:
            lines.append("  -> E not interpreted at this size")
            continue
        ready.append({"n": n, "p": p, "e_jump": e_jump, "agree": agree,
                      "hyst": p["hysteresis_met"], "jump": ascent_jump, "mean_jump": mean_j,
                      "split": split, "passing": passing})

    if len(ready) < 2:
        lines.append("")
        lines.append("NO VERDICT: T13's verdicts need two or more sizes that can be read; %d can."
                     % len(ready))
        return lines, "NO VERDICT"

    eq = [r for r in ready if r["e_jump"] and r["agree"] is True]
    meta = [r for r in ready if r["hyst"] and r["jump"] and not r["e_jump"]
            and r["agree"] is not False]

    lines.append("")
    if any(r["split"] for r in ready):
        lines.append("Note: the replica-mean reading differs from the majority reading at %s. "
                     "Amendment 3 makes the majority the verdict; the mean is above."
                     % ", ".join("N = %d" % r["n"] for r in ready if r["split"]))
    if len(eq) >= 2:
        lines.append("EQUILIBRIUM JUMP at %s" % ", ".join("N = %d" % r["n"] for r in eq))
        return lines, "EQUILIBRIUM JUMP"
    if len(meta) >= 2:
        lines.append("METASTABLE BRANCH at %s" % ", ".join("N = %d" % r["n"] for r in meta))
        return lines, "METASTABLE BRANCH"
    lines.append("INCONCLUSIVE: neither verdict's conditions are met at two or more sizes.")
    return lines, "INCONCLUSIVE"


def load(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main(ns, results="results"):
    sizes = {}
    for n in ns:
        s = {"p": None, "e": {}}
        try:
            s["p"] = read_p(load("%s/t13_seq_n%d.csv" % (results, n)))
        except FileNotFoundError:
            print("N = %d: no sequential run yet, skipped" % n)
            continue
        for start in ("melt", "torus"):
            try:
                s["e"][start] = read_e(load("%s/t13_temper_n%d_%s.csv" % (results, n, start)), n)
            except FileNotFoundError:
                pass
        if not s["e"]:
            print("N = %d: no tempering run yet, skipped" % n)
            continue
        sizes[n] = s
    if not sizes:
        return "NO VERDICT"
    lines, out = verdict(sizes)
    print("\n".join(lines))
    return out


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]] or [196, 484, 676])
