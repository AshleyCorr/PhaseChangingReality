"""Sandbox look, larger sample: 200 seeds x 100 burps per version, all seeds stepped in parallel.

Prediction (unchanged, written before the first run): crystal CV well under 1; popcorn CV near 1.
"""
import numpy as np

def popcorn(rng, seeds, n, p=1/110):
    return rng.geometric(p, size=(seeds, n)).astype(float)

def crystal(rng, seeds, n, gamma=0.1, noise=0.02, thr=4.0, dt=0.01, wander=0.0, grow=0.0):
    x = np.zeros(seeds); v = np.zeros(seeds); t = 0.0
    last = np.zeros(seeds); rs = np.zeros(seeds); g = np.full(seeds, gamma)
    ringing = np.zeros(seeds, bool)
    gaps = np.full((seeds, n), np.nan); count = np.zeros(seeds, int)
    sq = np.sqrt(dt)
    while (count < n).any():
        if wander:
            g = np.maximum(0.01, g + wander * rng.normal(size=seeds) * sq - 0.05 * (g - gamma) * dt)
        drive = g + grow * (t - last)
        damp = np.where(ringing, -0.3, drive)
        a = -x + damp * v + rng.normal(0, noise, seeds)
        v += a * dt; x += v * dt; t += dt
        e = 0.5 * (v * v + x * x)
        snap = (~ringing) & (e > thr) & (count < n)
        if snap.any():
            idx = np.where(snap)[0]
            gaps[idx, count[idx]] = t - last[idx]
            count[idx] += 1; last[idx] = t; rs[idx] = t; ringing[idx] = True
        reset = ringing & (t - rs > 20) & (e < 1e-3)
        ringing[reset] = False
    return gaps

def report(name, gaps, rng):
    cvs = gaps.std(axis=1) / gaps.mean(axis=1)
    boot = [rng.choice(cvs, cvs.size).mean() for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"{name:34s} mean gap {np.nanmean(gaps):7.1f}  CV {cvs.mean():.3f}  "
          f"95% range {lo:.3f}-{hi:.3f}  (single seeds {cvs.min():.2f}-{cvs.max():.2f})", flush=True)

if __name__ == "__main__":
    S, N = 200, 100
    r = np.random.default_rng(2026)
    report("A popcorn (Emily's memoryless)", popcorn(np.random.default_rng(1), S, N), r)
    report("B crystal, steady drive", crystal(np.random.default_rng(2), S, N), r)
    report("B' crystal, 5x noisier", crystal(np.random.default_rng(3), S, N, noise=0.1), r)
    report("C crystal, uneven feeding", crystal(np.random.default_rng(4), S, N, wander=0.05), r)
    report("D crystal, inner-horizon pileup", crystal(np.random.default_rng(5), S, N, gamma=0.02, grow=0.002), r)
