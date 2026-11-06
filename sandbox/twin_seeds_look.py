"""Sandbox look (5 November 2026, her words "maybe a seed is one the inside face of the torus and one on the
outside?"): on the packed 4 x L tube at T37's cell, the first two seeds of the escape. The tube's W columns
(y = 0..W-1, W = 4 unless given) are its faces; opposite faces are two apart round the 4-cycle. Recorded per replica: for each of the
first two seeds (T37's rule: a connected piece of at least `min_size` sheet points touching no seed counted before)
its sweep, its size, how many of the four faces it spans, its mean face (circular, round the 4-cycle) and its mean x
(circular, along the tube); then the face separation (0, 1 or 2) and the x separation of the two. By chance alone,
with each seed on a face drawn at random, the separation is 0, 1, 2 with weights 1/4, 1/2, 1/4. A look, not a test.
The wiring has no inside and no outside (every point of the torus is alike); the look can only ask whether the second
seed prefers the face opposite the first."""
import sys

import numpy as np

from graphity.cqg import NO_CAP, run_chain, torus
from graphity.dimension import local_dimension, new_seeds, piece_labels


def circ_mean(v, period):
    ang = 2 * np.pi * v / period
    return (np.arctan2(np.sin(ang).mean(), np.cos(ang).mean()) % (2 * np.pi)) * period / (2 * np.pi)


def circ_dist(a, b, period):
    d = abs(a - b) % period
    return min(d, period - d)


def one(L, rep, W=4, lam=1.25, g=1.5, block=10, max_sweeps=20000, min_size=8):
    adj, part = torus(L, W, NO_CAP)
    n = adj.shape[0]
    x_of, y_of = np.arange(n) // W, np.arange(n) % W
    side_u = np.flatnonzero(part == 0)
    seed = int(np.random.SeedSequence([20261105, 2, L, rep]).generate_state(1)[0])
    counted = np.zeros(n, dtype=bool)
    seeds, first = [], True
    for sweeps in range(0, max_sweeps, block):
        run_chain(adj, side_u, 1.0 / g, 0, block, seed if first else -1, lam=lam, cap=NO_CAP)
        first = False
        labels = piece_labels(adj, local_dimension(adj) == 2)
        if labels.max() < 0:
            continue
        sizes = np.bincount(labels[labels >= 0])
        fresh = [k for k in np.flatnonzero(sizes >= min_size) if not counted[labels == k].any()]
        n_new, counted = new_seeds(labels, counted, min_size)
        for k in fresh:
            m = labels == k
            seeds.append(dict(sweep=sweeps + block, size=int(m.sum()), faces=int(len(set(y_of[m]))),
                              face=circ_mean(y_of[m], W), x=circ_mean(x_of[m], L)))
        if len(seeds) >= 2:
            break
    return seeds


if __name__ == "__main__":
    L = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    reps = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    W = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    seps, xseps, spans, two = [], [], [], 0
    for rep in range(reps):
        s = one(L, rep, W)
        spans += [t["faces"] for t in s]
        line = f"W {W} L {L} rep {rep:2d}: " + " | ".join(
            f"seed {i + 1} sweep {t['sweep']} size {t['size']:3d} faces {t['faces']} face {t['face']:.1f} x {t['x']:.1f}"
            for i, t in enumerate(s))
        if len(s) >= 2:
            two += 1
            fs, xs = circ_dist(s[0]["face"], s[1]["face"], W), circ_dist(s[0]["x"], s[1]["x"], L)
            seps.append(fs)
            xseps.append(xs)
            line += f" | face sep {fs:.1f} x sep {xs:.1f}"
        print(line, flush=True)
    seps = np.array(seps)
    print(f"== W {W} L {L}: {two}/{reps} replicas gave two seeds; faces spanned by a seed: "
          f"{np.bincount(spans, minlength=W + 1)[1:]} (1..{W} faces); face separation of the first two seeds: "
          f"<0.5: {np.mean(seps < 0.5):.2f}, 0.5 to {W / 2 - 0.5}: {np.mean((seps >= 0.5) & (seps < W / 2 - 0.5)):.2f}, "
          f">={W / 2 - 0.5} (opposite): {np.mean(seps >= W / 2 - 0.5):.2f} (chance for W = 4: 0.25, 0.50, 0.25); "
          f"median x separation {np.median(xseps):.1f} links (uniform would be {L / 4:.1f})", flush=True)
