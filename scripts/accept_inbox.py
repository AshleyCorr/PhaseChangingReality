"""Check runs fetched into cloud/inbox/ against their committed configs, and move the ones that pass into results/.

    python scripts/accept_inbox.py            # check only; says what would move
    python scripts/accept_inbox.py --move     # check, then move every run that passes

The fetch_results workflow copies finished runs from the results bucket into cloud/inbox/<config>/. They are not
results until a session has checked them and committed the move (rule 5). A run passes when:

1. it has <config>.csv and <config>.meta.json;
2. the config recorded in its .meta.json is exactly configs/<config>.json as committed (every key and value);
3. its CSV parses and has a header and at least one row;
4. nothing of the same name is already in results/ (results are append-only; this script never overwrites).

Reported beside, not a gate: the number of rows, and the largest value of a `drift` column if the CSV has one (the
sealed runners' conservation check). A run that fails stays in the inbox, with the reason printed.
"""
import csv
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(run_dir, configs=ROOT / "configs", results=ROOT / "results"):
    """(ok, reasons, report) for one fetched run folder."""
    name = run_dir.name
    reasons, report = [], {}
    csv_path, meta_path = run_dir / (name + ".csv"), run_dir / (name + ".meta.json")
    if not csv_path.is_file() or not meta_path.is_file():
        return False, ["no CSV or no .meta.json"], report
    cfg_path = configs / (name + ".json")
    if not cfg_path.is_file():
        reasons.append("no committed config configs/%s.json" % name)
    else:
        recorded = json.loads(meta_path.read_text(encoding="utf-8")).get("config")
        if recorded != json.loads(cfg_path.read_text(encoding="utf-8")):
            reasons.append("the config in .meta.json differs from configs/%s.json" % name)
    with open(csv_path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        reasons.append("CSV has no rows")
    report["rows"] = len(rows)
    if rows and "drift" in rows[0]:
        report["max_drift"] = max(abs(float(r["drift"])) for r in rows if r["drift"] not in ("", None))
    for p in run_dir.iterdir():
        if (results / p.name).exists():
            reasons.append("results/%s already exists" % p.name)
    return not reasons, reasons, report


def move(run_dir, results=ROOT / "results"):
    for p in sorted(run_dir.iterdir()):
        shutil.move(str(p), str(results / p.name))
    run_dir.rmdir()


def main(argv):
    inbox = ROOT / "cloud" / "inbox"
    runs = sorted(p for p in inbox.iterdir() if p.is_dir()) if inbox.is_dir() else []
    if not runs:
        print("inbox empty")
        return 0
    passed = 0
    for run_dir in runs:
        ok, reasons, report = check(run_dir)
        extra = ", ".join("%s %s" % (k, ("%.2e" % v) if isinstance(v, float) else v) for k, v in report.items())
        print("%-4s %-40s %s%s" % ("OK" if ok else "FAIL", run_dir.name, extra, "" if ok else " | " + "; ".join(reasons)))
        if ok:
            passed += 1
            if "--move" in argv:
                move(run_dir)
    print("%d of %d pass%s" % (passed, len(runs), "; moved into results/" if "--move" in argv else ""))
    return 0 if passed == len(runs) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
