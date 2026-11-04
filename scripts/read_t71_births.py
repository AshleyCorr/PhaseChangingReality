"""T71's births one by one (TASKS, "26 October: the second Codex audit (O158)" item 1 and "the fourth Codex audit
(O159)" item 4). Usage, with the files named on the command line (never a glob):

    PYTHONPATH="src;scripts" python scripts/read_t71_births.py results/t71_model_lam110.csv [more .csv ...]

What it reads. A T71 result file logs, per replica, every sweep in which a move fired (`log_events`) and every
`log_every` sweeps besides. The daughter ledger `daughter_energy` only ever grows: each birth adds one daughter and
credits the lump to it. So a birth's energy is the rise in `daughter_energy` between consecutive logged rows, divided
by the number of daughters added in that row (always one in T71's files, O158 (a); a row adding more than one is
flagged). The first logged row counts: a replica whose first row already shows a daughter was born before or at that
row, and the rise is taken from zero. This is the count `analyse_t71.read_replica` took only for the period, and the
count Codex made (O158 addendum, F31): 5,148 and 4,620 births in the clean-switch cells against the scratch script's
5,132 and 4,604, one per replica.

Against what. Each birth's energy is reported against a N_slab, a = 4(lambda - 1) per point, the untied release of
the slab's one curled direction (VISION Update 69, gate g1). lambda and N_slab are read from each file's rows (the
`.meta.json` beside the CSV is read too and checked for a tie; T71's configs are untied, and if a meta names a tie
the reader says so and does not pretend to know the tied release).

What is counted, per replica and summed per cell (lam, L, rate, move_set, release_to): births; births exactly a N_slab
(within 1e-9); births below 0.99 a N_slab, with the shortest; births above a N_slab by any amount, with the largest
ratio; births in (1, 1.25] a N_slab; births above 1.25 a N_slab; and refolds between consecutive births, read as
`n_contract` rising between the two birth rows (the reading of O158 (a)). The task text says "daughter_energy
falling"; that ledger cannot fall, so a fall is reported as an anomaly count and should be zero.

Starting state (O159 item 1): T71's starting X is the slab 4 x L x L, six links, one direction curled to length 4.
Exact: a direction of length 4 closes extra squares (in two dimensions the tube carries 1.25 squares per point
against the sheet's 1), so the slab holds more than 3N squares and does not obey the hard rule the owner adopted on
26 October (squares held at exactly 3N for six links). This reader reads a registered run as it was run; it is not a
look and does not score anything. T71's held-out score stays `t71_heldout_score.py`.

Everything printed is exact arithmetic on the logged rows.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

EXACT = 1e-9
BELOW = 0.99
ABOVE_BAND = 1.25


def tie_of(meta_path):
    """The tie named in a file's meta, or None when the meta is absent or names no tie."""
    p = Path(meta_path)
    if not p.exists():
        return None
    cfg = json.loads(p.read_text()).get("config", {})
    tie = cfg.get("tie", cfg.get("tie_form"))
    if tie in (None, "", "none", "untied", False):
        return None
    return tie


def births_of(rows):
    """rows: one replica's logged rows. Returns (births, refolds, anomalies, flagged) where births is a list of
    (sweep, energy) with the first logged row counted, refolds the number of consecutive birth pairs with n_contract
    rising between them, anomalies the number of rows where daughter_energy fell, flagged the rows adding > 1."""
    rows = sorted(rows, key=lambda r: int(r["sweep"]))
    births, flagged, anomalies = [], 0, 0
    prev_d, prev_e = 0, 0.0
    contract_at_birth = []
    for r in rows:
        d, e = int(r["daughters"]), float(r["daughter_energy"])
        if e < prev_e - EXACT:
            anomalies += 1
        k = d - prev_d
        if k > 0:
            if k > 1:
                flagged += 1
            births.extend([(int(r["sweep"]), (e - prev_e) / k)] * k)
            contract_at_birth.extend([int(r["n_contract"])] * k)
        prev_d, prev_e = d, e
    refolds = sum(1 for a, b in zip(contract_at_birth[:-1], contract_at_birth[1:]) if b > a)
    return births, refolds, anomalies, flagged


def read_replica(rows):
    r0 = sorted(rows, key=lambda r: int(r["sweep"]))[0]
    lam, n_slab = float(r0["lam"]), int(r0["N_slab"])
    a_n = 4.0 * (lam - 1.0) * n_slab
    births, refolds, anomalies, flagged = births_of(rows)
    ratios = [e / a_n for _, e in births]
    return dict(
        a_n=a_n, births=len(births), energies=[e for _, e in births], ratios=ratios,
        exact=sum(1 for x in ratios if abs(x - 1.0) <= EXACT / a_n),
        below=sum(1 for x in ratios if x < BELOW), shortest=min(ratios) if ratios else None,
        above=sum(1 for x in ratios if x > 1.0 + EXACT / a_n), largest=max(ratios) if ratios else None,
        above_band=sum(1 for x in ratios if 1.0 + EXACT / a_n < x <= ABOVE_BAND),
        above_125=sum(1 for x in ratios if x > ABOVE_BAND),
        refolds=refolds, pairs=max(len(births) - 1, 0), anomalies=anomalies, flagged=flagged)


def read_cells(rows):
    groups = defaultdict(lambda: defaultdict(list))
    for r in rows:
        key = (float(r["lam"]), int(r["L"]), float(r["rate"]), r["move_set"], r.get("release_to", "stores"))
        groups[key][int(r["replica"])].append(r)
    cells = {}
    for key, reps in sorted(groups.items()):
        reads = [read_replica(rs) for _, rs in sorted(reps.items())]
        tot = lambda f: sum(x[f] for x in reads)  # noqa: E731
        shortest = [x["shortest"] for x in reads if x["shortest"] is not None]
        largest = [x["largest"] for x in reads if x["largest"] is not None]
        cells[key] = dict(
            n=len(reads), a_n=reads[0]["a_n"], births=tot("births"), exact=tot("exact"), below=tot("below"),
            shortest=min(shortest) if shortest else None, above=tot("above"), largest=max(largest) if largest else None,
            above_band=tot("above_band"), above_125=tot("above_125"), refolds=tot("refolds"), pairs=tot("pairs"),
            anomalies=tot("anomalies"), flagged=tot("flagged"),
            first_row_births=sum(1 for _, rs in sorted(reps.items())
                                 if int(sorted(rs, key=lambda r: int(r["sweep"]))[0]["daughters"]) > 0),
            replicas=reads)
    return cells


def main(paths):
    if not paths:
        raise SystemExit("name the T71 result files on the command line (no glob is used)")
    rows = []
    for p in paths:
        tie = tie_of(str(p)[:-4] + ".meta.json" if str(p).endswith(".csv") else str(p) + ".meta.json")
        if tie is not None:
            print("%s: meta names a tie (%r); a N_slab is the untied release and is not the tied one" % (p, tie))
        rows.extend(csv.DictReader(open(p, newline="")))
    cells = read_cells(rows)
    for (lam, L, rate, move_set, release_to), c in cells.items():
        print("lam=%.2f L=%2d r=%-7g %-8s %-8s | replicas %d | births %d (%d counted at a first logged row) | "
              "exactly a N_slab %d | below 0.99: %d, shortest %s | above: %d, largest %s | in (1, 1.25]: %d | "
              "above 1.25: %d | refolds between consecutive births %d of %d pairs | ledger falls %d | multi-birth rows %d"
              % (lam, L, rate, move_set, release_to, c["n"], c["births"], c["first_row_births"], c["exact"],
                 c["below"], None if c["shortest"] is None else "%.3f" % c["shortest"], c["above"],
                 None if c["largest"] is None else "%.3f" % c["largest"], c["above_band"], c["above_125"],
                 c["refolds"], c["pairs"], c["anomalies"], c["flagged"]))
    return cells


if __name__ == "__main__":
    main(sys.argv[1:])
