"""Sandbox look: an impure crystal X. Do its flaws decide where it snaps, give a direction, and keep the rhythm?
(6 Nov 2026, Ashley's idea: the squeeze makes a crystal that is not pure)

Predictions written before running:
  P1 Snaps follow the flaws: the snap lands on or next to a flaw far more often than the flaws' share of area.
  P2 Spin pattern (flaws along the equator band of a spinning black hole): snaps line up with that band
     (alignment near 1); random and evenly spaced flaws give no direction (near 0).
  P3 Layers (rings of flaws, the innermost being the leftover of the last burp): snaps sit at the ring radii.
  P4 The rhythm survives: spread of gaps stays well under 1 for every pattern.

Model: the drumhead disk of hotspot_look.py. A flaw is a site that gives way at a lower energy (0.7 of the
threshold). Each crystal keeps its flaws and snaps 10 times (reset to quiet noise after each snap).
Alignment = average of cos(2 x angle of the snap point): 1 = along the band, 0 = no direction.
"""
import numpy as np

R = 15; N = 2 * R + 3
YY, XX = np.mgrid[:N, :N] - (N - 1) / 2
RR = np.hypot(XX, YY); TH = np.arctan2(YY, XX)
DISK = RR <= R

def flaws(rng, kind):
    f = np.zeros((N, N), bool)
    if kind == "none":
        return f
    if kind == "random":
        f = DISK & (rng.random((N, N)) < 0.03)
    elif kind in ("spaced_slow", "spaced_fast"):
        s = 7 if kind == "spaced_slow" else 4                 # faster squeeze -> closer flaws (Kibble-Zurek)
        ox, oy = rng.integers(0, s, 2)
        f = DISK & ((XX + R + ox).astype(int) % s == 0) & ((YY + R + oy).astype(int) % s == 0)
    elif kind == "spin":
        f = DISK & (np.abs(YY) <= 1.5) & (rng.random((N, N)) < 0.25)   # equator band, spin axis fixed
    elif kind == "layers":
        f = DISK & np.isin(np.round(RR).astype(int), [4, 9, 13]) & (rng.random((N, N)) < 0.35)
    return f

def crystal(rng, f, burps=10, c=1.0, k=1.0, pump=0.2, noise=0.02, thr=4.0, dt=0.05):
    th = np.where(f, 0.7 * thr, thr)
    near = f | np.roll(f, 1, 0) | np.roll(f, -1, 0) | np.roll(f, 1, 1) | np.roll(f, -1, 1)
    out, t, last = [], 0.0, 0.0
    u = np.zeros((N, N)); v = np.zeros((N, N))
    while len(out) < burps:
        lap = np.roll(u, 1, 0) + np.roll(u, -1, 0) + np.roll(u, 1, 1) + np.roll(u, -1, 1) - 4 * u
        a = c * c * lap - k * u + pump * v + noise * rng.normal(size=u.shape) / np.sqrt(dt)
        v += a * dt; v *= DISK; u += v * dt; u *= DISK; t += dt
        over = (0.5 * (v * v + k * u * u)) / th
        if over.max() > 1:
            i = np.unravel_index(over.argmax(), over.shape)
            out.append((t - last, RR[i] / R, TH[i], near[i])); last = t
            u = 0.001 * rng.normal(size=u.shape) * DISK; v[:] = 0
    return out

if __name__ == "__main__":
    S = 100
    print(f"{'pattern':14s} {'on/next to flaw':>16s} {'flaw area':>10s} {'alignment':>10s} {'mean dist':>10s} {'gap spread':>11s}")
    for kind in ["none", "random", "spaced_slow", "spaced_fast", "spin", "layers"]:
        rows, area, cvs = [], [], []
        for s in range(S):
            rng = np.random.default_rng(1000 * hash(kind) % 99991 + s)
            f = flaws(rng, kind)
            nearf = f | np.roll(f, 1, 0) | np.roll(f, -1, 0) | np.roll(f, 1, 1) | np.roll(f, -1, 1)
            area.append((nearf & DISK).sum() / DISK.sum())
            out = crystal(rng, f); rows += out
            g = np.array([o[0] for o in out]); cvs.append(g.std() / g.mean())
        d = np.array([o[1] for o in rows]); ang = np.array([o[2] for o in rows]); hit = np.array([o[3] for o in rows])
        print(f"{kind:14s} {hit.mean():16.2f} {np.mean(area):10.2f} {np.mean(np.cos(2 * ang)):10.2f} "
              f"{d.mean():10.2f} {np.mean(cvs):11.2f}", flush=True)
    if True:
        rng = np.random.default_rng(7)
        for kind in ["layers"]:
            ds = []
            for s in range(S):
                f = flaws(rng, kind); ds += [o[1] * R for o in crystal(rng, f)]
            h = np.bincount(np.round(ds).astype(int), minlength=R + 1)
            print("layers: snaps by radius 0..15:", " ".join(str(x) for x in h[:R + 1]), "(rings at 4, 9, 13)")
