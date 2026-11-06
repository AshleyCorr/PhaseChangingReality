"""Sandbox look: where does a vibrating crystal X snap? (6 Nov 2026, Ashley)

Prediction written before running (Ashley's idea, our wording): a vibrating crystal has hot spots, so it snaps
off-centre more often than chance would put it, and the snap point sits away from the middle.
Chance baseline (Emily's still X stand-in, snap at a random site, uniform over the sheet): mean distance from
centre 0.67 R, share of snaps beyond half the radius 0.75.
A result near or below the baseline means the crystal does NOT favour off-centre snaps.

Model: a flat disk of sites (radius R) joined to neighbours like a drumhead, each also springy on its own;
noise everywhere; a uniform pump (negative damping) amplifies every wave equally; the sheet snaps at the first
site whose local energy crosses a threshold. Two shapes: a full disk, and a doughnut (hole of 0.4 R in the middle,
Emily's rod) where "off-centre" is read as distance from the rod.
"""
import numpy as np

def run(rng, R=15, hole=0.0, c=1.0, k=1.0, pump=0.2, noise=0.02, thr=4.0, dt=0.05):
    n = 2 * R + 3
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    r = np.hypot(xx, yy)
    mask = (r <= R) & (r >= hole * R)
    u = np.zeros((n, n)); v = np.zeros((n, n))
    for _ in range(200000):
        lap = (np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u)
        a = c * c * lap - k * u + pump * v + noise * rng.normal(size=u.shape) / np.sqrt(dt)
        v += a * dt; v *= mask; u += v * dt; u *= mask
        e = 0.5 * (v * v + k * u * u)
        if e.max() > thr:
            i = np.unravel_index(e.argmax(), e.shape)
            return r[i] / R
    return np.nan

def baseline(rng, R=15, hole=0.0, m=20000):
    n = 2 * R + 3
    yy, xx = np.mgrid[:n, :n] - (n - 1) / 2
    r = np.hypot(xx, yy)
    pts = r[(r <= R) & (r >= hole * R)]
    return rng.choice(pts, m) / R

def show(name, d):
    d = d[~np.isnan(d)]
    boot = [np.random.default_rng(i).choice(d, d.size).mean() for i in range(1000)]
    print(f"{name:40s} mean distance {d.mean():.2f} R  (95% {np.percentile(boot,2.5):.2f}-{np.percentile(boot,97.5):.2f})"
          f"   beyond half radius {np.mean(d > 0.5):.2f}   in the middle fifth {np.mean(d < 0.2):.2f}", flush=True)

if __name__ == "__main__":
    S = 200
    show("Still X, random site (disk)", baseline(np.random.default_rng(1)))
    show("Crystal, full disk", np.array([run(np.random.default_rng(100 + s)) for s in range(S)]))
    show("Crystal, full disk, stiffer sheet (c=2)", np.array([run(np.random.default_rng(300 + s), c=2.0) for s in range(S)]))
    show("Still X, random site (doughnut)", baseline(np.random.default_rng(2), hole=0.4))
    show("Crystal, doughnut round the rod", np.array([run(np.random.default_rng(500 + s), hole=0.4) for s in range(S)]))
