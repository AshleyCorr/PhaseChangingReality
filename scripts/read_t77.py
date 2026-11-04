"""Read results/t77_diamond_w.csv as T73 read its rows: per (w, class), the scored verdict and the spread over the
sensitivity rows (delay, n, tail, cap) with the words that occur. Prints a table; writes nothing.

Usage: python scripts/read_t77.py results/t77_diamond_w.csv
"""
import csv
import sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))


def f(x):
    return float(x) if x not in ("", None) else None


by = defaultdict(list)
for r in rows:
    by[(float(r["w"]), r["cls"])].append(r)
for (w, cls), rs in sorted(by.items(), key=lambda kv: (-kv[0][0], kv[0][1])):
    scored = [r for r in rs if r["scored"] == "True"]
    Fs = [f(r["F_now"]) for r in rs]
    words = sorted(set(r["verdict"] for r in rs))
    s = scored[0]
    print(f"w={w:+.3f} {cls:<4} SCORED F={f(s['F_now']):.4f} peak={f(s['peak_dPdt_Gyr']):.2f} {s['verdict']:<8} "
          f"| sensitivities F {min(Fs):.4f} to {max(Fs):.4f}, words {words}")
