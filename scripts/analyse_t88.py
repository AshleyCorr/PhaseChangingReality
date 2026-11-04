"""T88, the pole test: read the verdict words from the rows run_rod_pole.py wrote. Usage:

    python scripts/analyse_t88.py [results_dir]            # every results/t88_*.csv
    python scripts/analyse_t88.py results --held-out        # prints the L = 16 prediction fitted on L = 8 and 12

Definitions, fixed in PREREGISTRATION.md section T88 before any run:
  opened        some reading has a connected region of at least MIN_REGION (16) points at d = 3.
  t_first       the first such reading's sweep.
  from the end  at t_first the fed layer (z = 0) holds at least one point at d = 3.
  t_far         the first sweep at which the far layer (z = L / 2) holds at least 8 of its 16 points at d = 3;
                None if never.
  ONE FRONT     opened, from the end, and open_regions3 <= 1 at every reading up to t_far (to the end if never).
  SEVERAL       opened, but not from the end, or two open regions at some reading before t_far.
  NOTHING OPENS not opened.
A cell's word is the majority word of its replicas; a tie goes to SEVERAL. Damage (d > 3) is reported, not scored.
The held-out prediction: t_far's cell mean at L = 16 is the straight-line extrapolation of the cell means at L = 8 and
12 (a front at a fixed speed makes t_far grow with the distance to the far side), with a range of twice the larger of
the two cells' standard errors; the cell word at L = 16 is predicted as the word the two smaller sizes share.
"""
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

MIN_REGION = 16
FAR_OPEN = 8


def rows_of(path):
    with open(path, newline="") as fh:
        return [dict(r) for r in csv.DictReader(fh)]


def replica_word(rows):
    rows = sorted(rows, key=lambda r: int(r["sweep"]))
    t_first = next((int(r["sweep"]) for r in rows if int(r["largest_open3"]) >= MIN_REGION), None)
    if t_first is None:
        return "NOTHING OPENS", None, None
    first = next(r for r in rows if int(r["sweep"]) == t_first)
    from_end = int(first["fed_open3"]) >= 1
    t_far = next((int(r["sweep"]) for r in rows if int(r["far_open3"]) >= FAR_OPEN), None)
    upto = [r for r in rows if t_far is None or int(r["sweep"]) <= t_far]
    one_region = all(int(r["open_regions3"]) <= 1 for r in upto)
    word = "ONE FRONT" if (from_end and one_region) else "SEVERAL"
    return word, t_first, t_far


def cell_word(words):
    c = Counter(words)
    top = max(c.values())
    leaders = [w for w, k in c.items() if k == top]
    return leaders[0] if len(leaders) == 1 else "SEVERAL"


def mean_se(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return None, None, 0
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, None, len(xs)
    v = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, (v / len(xs)) ** 0.5, len(xs)


def analyse(results_dir="results", held_out=False):
    cells = defaultdict(lambda: defaultdict(list))
    damage = defaultdict(list)
    for p in sorted(Path(results_dir).glob("t88_*.csv")):
        by_rep = defaultdict(list)
        for r in rows_of(p):
            by_rep[int(r["replica"])].append(r)
        for rep, rows in by_rep.items():
            length = int(rows[0]["dims"].split()[-1])
            key = (float(rows[0]["lam"]), float(rows[0]["energy"]), length)
            word, t_first, t_far = replica_word(rows)
            cells[key]["words"].append(word)
            cells[key]["t_far"].append(t_far)
            last = max(rows, key=lambda r: int(r["sweep"]))
            damage[key].append(int(last["damaged"]))
    out = {}
    for key in sorted(cells):
        words = cells[key]["words"]
        m, se, k = mean_se(cells[key]["t_far"])
        out[key] = dict(word=cell_word(words), words=Counter(words), t_far_mean=m, t_far_se=se, n_far=k,
                        damaged_mean=sum(damage[key]) / len(damage[key]))
        print("lambda=%.2f E=%g L=%-3d | %-13s %s | t_far %s (se %s, %d reps reached) | damaged at end %.1f"
              % (key[0], key[1], key[2], out[key]["word"], dict(out[key]["words"]),
                 "%.0f" % m if m is not None else "never", "%.0f" % se if se else "-", k, out[key]["damaged_mean"]))
    if held_out:
        print("\nHeld-out prediction for L = 16 (fitted on L = 8 and 12; written before L = 16 runs):")
        for lam, e in sorted({(k[0], k[1]) for k in out}):
            a, b = out.get((lam, e, 8)), out.get((lam, e, 12))
            if not a or not b:
                continue
            word = a["word"] if a["word"] == b["word"] else "UNDECIDED (the two sizes differ: %s, %s)" % (a["word"], b["word"])
            if a["t_far_mean"] is not None and b["t_far_mean"] is not None:
                pred = 2 * b["t_far_mean"] - a["t_far_mean"]
                half = 2 * max(a["t_far_se"] or 0.0, b["t_far_se"] or 0.0)
                tf = "t_far %.0f, range %.0f to %.0f" % (pred, pred - half, pred + half)
            else:
                tf = "t_far: never (no fit)"
            print("  lambda=%.2f E=%g: word %s; %s" % (lam, e, word, tf))
    return out


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    analyse(args[0] if args else "results", held_out="--held-out" in sys.argv)
