"""Retrospective T24 residual-wait diagnostic on T60; no new registration verdict.

Run from the repository root. Calibration uses the recorded exponential reference
seed, 200,000 samples, and the actual number of eligible waits in each cell.
"""
import csv
from functools import lru_cache
import json
from pathlib import Path
import numpy as np


@lru_cache(None)
def band(n):
    rng = np.random.default_rng(20262399)
    samples = rng.exponential(size=(200000, n))
    return np.quantile(samples.std(axis=1, ddof=1)/samples.mean(axis=1), [.0005, .9995])


def read(root):
    cells = []
    for path in sorted((root/'results').glob('t60_lam*_n*.csv')):
        with path.open() as stream:
            rows = list(csv.DictReader(stream))
        waits = np.array([float(r['waiting'])-200 for r in rows
                          if float(r['reached'] or 0) >= .75 and r['waiting']])
        cv = float(waits.std(ddof=1)/waits.mean())
        interval = band(len(waits))
        cells.append(dict(N=int(rows[0]['N']), lam=float(rows[0]['lam']), n_old=len(waits),
                          old_cv=cv, old_band=interval.tolist(),
                          old_pass=bool(interval[0] <= cv <= interval[1])))
    return cells


if __name__ == '__main__':
    print(json.dumps(read(Path(__file__).resolve().parents[1]), indent=2))
