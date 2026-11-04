"""Write T68's configs (PREREGISTRATION T68, written 2026-10-17 on the owner's "proceed with these", before any
registered run): the owner's untied slab, 4 x L x L with one direction curled, sealed, fed under the two variants T67
named as untried (a rate that varies; a feed arriving at many points), at three curling costs. Usage:

    python scripts/make_t68_configs.py              # stage 1 (L = 8 and 12 in one job); prints the queue lines
    python scripts/make_t68_configs.py --held-out   # L = 16, after the smaller sizes are read (rule 14)

Refuses to rewrite an existing config. One job per (schedule, place, lambda): schedules fixed (the control), ramp and
pulse; places shared (store 0 of a bath of N stores), spot (one point), spots (16 points), all (every point); lambda
1.25, 1.40 and 1.60; mean rates 1/16, 1/4 and 1 units a sweep; 8 replicas; up to 20,000 sweeps read every 50; the
final graph saved. Seeds from 20266901 in the order written (held-out from 20267001).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEED0 = 20266901
RATES = [0.0625, 0.25, 1.0]
LAMBDAS = [1.25, 1.40, 1.60]
SCHEDULES = ["fixed", "ramp", "pulse"]
PLACES = [("shared", "store 0 of a shared bath of N stores"), ("spot", "one point's own store"),
          ("spots", "16 points' own stores, chosen at random"), ("all", "every point's own store")]
K_SPOTS = 16
PURPOSE = ("PREREGISTRATION.md T68 (written 2026-10-17 on the owner's 'proceed with these', before any registered run; "
           "rule 11's sixth use on piece 5; VISION Updates 58 and 59): does the owner's untied slab (4 x L x L, one "
           "direction curled) open from a feed whose rate %s, arriving at %s? lambda %s; mean rates %s units a sweep. "
           "Not exploratory. Every six-link result carries VISION Update 24's caveat.")
SCHED_WORDS = {"fixed": "is fixed (the control)", "ramp": "rises linearly from zero to twice the mean over the run",
               "pulse": "arrives as bursts of 128 units at the mean rate"}


def cells(held_out=False):
    sizes = [16] if held_out else [8, 12]
    out = []
    for lam in LAMBDAS:
        for sched in SCHEDULES:
            for place, label in PLACES:
                name = "t68_%s_%s_lam%d%s" % (sched, place, round(100 * lam), "_l16" if held_out else "")
                out.append((name, sizes, sched, place, lam, label))
    return out


def config(name, sizes, sched, place, lam, label, seed):
    return {"_purpose": PURPOSE % (SCHED_WORDS[sched], label, lam, RATES), "name": name, "section": "T68",
            "written": "2026-10-17", "sizes": sizes, "lambda": lam, "mode": place, "schedule": sched, "k": K_SPOTS,
            "capacity": "N", "rates": RATES, "replicas": 8, "n_sweeps": 20000, "record_every": 50, "seed": seed,
            "save_adjacency": True}


def main(held_out=False):
    offset = 100 if held_out else 0
    for i, (name, sizes, sched, place, lam, label) in enumerate(cells(held_out)):
        path = ROOT / "configs" / (name + ".json")
        if path.exists():
            print("exists, not rewritten:", path.name, file=sys.stderr)
        else:
            path.write_text(json.dumps(config(name, sizes, sched, place, lam, label, SEED0 + offset + i), indent=2) + "\n", encoding="utf-8")
        print("run_slab_feed", name)


if __name__ == "__main__":
    main("--held-out" in sys.argv[1:])
