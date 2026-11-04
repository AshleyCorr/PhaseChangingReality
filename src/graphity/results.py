"""Writing results without ever overwriting them (README reproducibility rule 2).

Rows are streamed to results/<name>.csv.partial as they are produced, so a crash
keeps everything that had finished. Only a run that completes is renamed to
results/<name>.csv, with <name>.meta.json written beside it. If either final file
already exists the run refuses to start: to run a config again, copy it under a
new name. A leftover .partial file is scratch from a crashed run; it is ignored by
git and overwritten by the next attempt.
"""
import csv
import json
from pathlib import Path


class ResultWriter:
    """Use as:   with ResultWriter(name, meta) as out: out.write(row_dict)"""

    def __init__(self, name, meta, out_dir="results"):
        self.csv_path = Path(out_dir) / f"{name}.csv"
        self.meta_path = self.csv_path.with_suffix(".meta.json")
        self._refuse_if_present()
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.meta = meta
        self.partial_path = self.csv_path.with_name(self.csv_path.name + ".partial")
        self._fh = self.partial_path.open("w", newline="")
        self._writer = None

    def _refuse_if_present(self):
        for p in (self.csv_path, self.meta_path):
            if p.exists():
                raise FileExistsError(
                    f"{p} already exists. Results are append-only: to run this again, "
                    "copy the config under a new name.")

    def write(self, row):
        if self._writer is None:                 # columns are fixed by the first row
            self._writer = csv.DictWriter(self._fh, fieldnames=list(row.keys()))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._fh.flush()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self._fh.close()
        if exc_type is None:                     # a failed run stays a .partial file
            self._refuse_if_present()
            self.partial_path.rename(self.csv_path)
            self.meta_path.write_text(json.dumps(self.meta, indent=2))
        return False


def provenance(root=None, max_listed=20):
    """The commit the code ran from and whether the working tree differed from it, for a run's .meta.json
    (TASKS, the reviews of 6 October, item 5). Outside a git checkout, or without git, the commit is "unknown".

    The tree counts as dirty when any tracked file is modified, staged or deleted, or when any file that git does
    not ignore is untracked: code that has never been committed cannot be recovered from the commit named here,
    so an untracked file is as much a departure from the commit as an edited one. (Until 2026-10-12 untracked files
    were not looked at, which let three results of that day name a commit that held none of their code; the
    provenance note in docs/theory/orh_findings.md records it.) `git_untracked_count` says how many such files there
    were and `git_untracked` lists the first `max_listed` of them, so a reader can see why the tree was dirty."""
    import subprocess
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
                                check=True).stdout.strip()
        status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=root,
                                capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return dict(git_commit="unknown", git_dirty=None, git_untracked_count=None, git_untracked=None)
    lines = [l for l in status.splitlines() if l.strip()]
    untracked = sorted(l[3:] for l in lines if l.startswith("??"))      # ignored files never appear here
    return dict(git_commit=commit, git_dirty=bool(lines), git_untracked_count=len(untracked),
                git_untracked=untracked[:max_listed])
