"""Read results/t72_observer_window.csv as T72 registers: scored rows (delay as registered, n = 5, tail 50 Gyr, the
class's own cap setting) and the sensitivity spread per class. Prints a table; writes nothing."""
import csv, sys
from collections import defaultdict

rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8")))
def f(x): return float(x) if x not in ("", None) else None
by = defaultdict(list)
for r in rows:
    by[r["cls"]].append(r)
for cls in ["gate", "A", "B", "C", "D"]:
    rs = by[cls]
    if not rs:
        continue
    scored = [r for r in rs if r["scored"] == "True"]
    print(f"\n== class {cls}: {len(rs)} rows, {len(scored)} scored")
    for r in scored:
        print(f"  SCORED  F_now={f(r['F_now']):.4f}  peak dP/dt={f(r['peak_dPdt_Gyr']):.2f} Gyr  "
              f"peak dP/dlnt={f(r['peak_dPdlnt_Gyr']):.1f}  median={f(r['median_Gyr']):.1f}  {r['verdict']}")
    Fs = [f(r["F_now"]) for r in rs]
    peaks = [f(r["peak_dPdt_Gyr"]) for r in rs]
    words = sorted(set(r["verdict"] for r in rs))
    print(f"  sensitivity: F_now {min(Fs):.4f} to {max(Fs):.4f}; peak dP/dt {min(peaks):.2f} to {max(peaks):.2f} Gyr; words {words}")
    for r in rs:
        print(f"    delay={r['delay_Gyr']} n={r['n_steps']} tail={r['tail_Gyr']} cap={r['cap_Gyr']}: F={f(r['F_now']):.4f} peak={f(r['peak_dPdt_Gyr']):.2f} {r['verdict']}")
