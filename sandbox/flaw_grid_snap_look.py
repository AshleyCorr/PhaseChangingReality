"""Sandbox look (18 November 2026, Ashley's flaw-grid snap; rule 16: a look, not a test, nothing registered).
A standalone grid, NOT the toy (graphity): an energy-fill-and-release cellular model of his "crystal X with
flaws", read on a sphere. It can kill but not crown. Output goes to sandbox/out/ (untracked).

Flaw-grid snap simulation
=========================

Idea being tested
-----------------
Crystal X stores energy until a threshold is reached, then "snaps".
Like the pinning sites in a flux-pinned superconductor puck, its FLAWS are
where release starts. The snap spreads outward from the flaws, so some regions
snap earlier than others. If earlier-snapping regions expand a little more,
the resulting universe is slightly uneven.

Question: how does a flawed crystal spread its unevenness across big and
small patches of sky, and does that match the real CMB (roughly
scale-invariant)? Lopsidedness A is kept as a secondary measure.

Main measure: large_scale_fraction = share of the unevenness in the
biggest patterns (multipoles l=2-4) out of l=2-20. A scale-invariant sky,
which the real CMB roughly is on these scales, gives the reference value.

How it works
------------
1. Build an N x N x N crystal. Every cell has a snap threshold of 1.0.
   A fraction FLAW_FRACTION of cells are flaws with a lower threshold
   (1 - FLAW_DEPTH). Flaws can be scattered or clumped (FLAW_CLUMPING).
2. Fill the crystal with energy slowly and evenly (tiny random noise).
3. When a cell passes its threshold it snaps and releases its stored energy
   (RELEASE) to its 6 neighbours, which can set off a cascade (the snap front).
4. Record WHEN each cell snapped. Unevenness field = normalised snap time.
5. Read that field on a sphere inside the crystal (our "sky"), and measure
   the hemispherical asymmetry amplitude A — the same kind of number Planck
   reports for the real CMB.
6. Repeat many times for flawed and for perfect (null) crystals.

Honest limits
-------------
- This is a statistical toy, not real cosmology: it doesn't model inflation,
  plasma physics or how snap timing becomes a temperature pattern.
- The A comparison is "same kind of number", not an exact like-for-like
  with Planck's pipeline. A real comparison needs the Planck map + healpy
  (see compare_to_planck() at the bottom).

Before any real data
--------------------
The script writes its numbers to sandbox/out/ with a timestamp. If this is
ever promoted, the prediction goes in PREREGISTRATION.md (rule 4) before
compare_to_planck() is run; this look does not count as that.
"""

import json
import time
from pathlib import Path
import numpy as np
from scipy import ndimage

# ---------------- parameters (change these, then re-run) ----------------
N = 64                 # crystal size per side (cells)
FLAW_FRACTION = 0.002  # fraction of cells that are flaws
FLAW_DEPTH = 0.15      # how much lower a flaw's threshold is
FLAW_CLUMPING = 4.0    # 0 = scattered flaws; >0 = flaws cluster (smoothing scale, cells)
RELEASE = 0.9          # fraction of its stored energy a snapping cell releases to its 6 neighbours
FILL_STEP = 0.002      # energy added per time step
FRONT_SPEED = 64       # cells the snap front can travel per time step (lower = slower front,
                       # so more flaws go off on their own before a front reaches them)
NOISE = 0.01           # random unevenness in the energy fill
SHELL_RADIUS = 0.35    # sky sphere radius as a fraction of N
N_SKY = 3072           # points on the sky sphere
N_RUNS = 15            # simulations per scenario
SEED = 2026
PLANCK_A_REFERENCE = 0.07  # Planck large-scale asymmetry, general knowledge, to verify (Planck 2018 VII)


def make_thresholds(rng, flawed: bool):
    thr = np.ones((N, N, N))
    if not flawed:
        return thr
    n_flaws = max(1, int(FLAW_FRACTION * N**3))
    if FLAW_CLUMPING > 0:
        # clumped flaws: pick flaw sites where a smooth random field is highest
        field = ndimage.gaussian_filter(rng.standard_normal((N, N, N)),
                                        FLAW_CLUMPING, mode="wrap")
        field += 0.5 * rng.standard_normal((N, N, N))
        idx = np.argpartition(field.ravel(), -n_flaws)[-n_flaws:]
    else:
        idx = rng.choice(N**3, n_flaws, replace=False)
    thr.ravel()[idx] = 1.0 - FLAW_DEPTH
    return thr


def run_snap(thr, rng):
    """Fill with energy; return the time step at which each cell snapped."""
    hop = 1.0 / FRONT_SPEED
    energy = rng.uniform(0, NOISE, thr.shape)
    snapped = np.zeros(thr.shape, bool)
    snap_t = np.full(thr.shape, np.nan)
    t = 0
    while not snapped.all():
        t += 1
        energy[~snapped] += FILL_STEP * (1 + NOISE * rng.standard_normal(thr.shape))[~snapped]
        # cascade: the front moves at most FRONT_SPEED cells this step;
        # energy already passed on carries over and continues next step.
        g = 0
        while g < FRONT_SPEED:
            new = (~snapped) & (energy >= thr)
            if not new.any():
                break
            snapped |= new
            snap_t[new] = t + g * hop
            g += 1
            k = np.where(new, energy * RELEASE / 6, 0.0)
            energy[new] = 0.0
            for ax in range(3):
                energy += np.roll(k, 1, ax) + np.roll(k, -1, ax)
        if t > 10_000:
            raise RuntimeError("did not finish — check parameters")
    return snap_t


def sky_points(n):
    """Evenly spread points on a sphere (Fibonacci lattice)."""
    i = np.arange(n) + 0.5
    z = 1 - 2 * i / n
    phi = np.pi * (1 + 5**0.5) * i
    r = np.sqrt(1 - z**2)
    return np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=1)


SKY = sky_points(N_SKY)


def read_sky(field):
    """Sample the unevenness field on a sphere centred in the crystal."""
    c = (N - 1) / 2
    coords = (c + SHELL_RADIUS * N * SKY).T
    field = ndimage.gaussian_filter(field, 1.5, mode="wrap")  # expansion smooths single-cell spikes
    vals = ndimage.map_coordinates(field, coords, order=1, mode="wrap")
    return (vals - vals.mean()) / vals.std()


def asymmetry_A(sky_vals, cap_deg=30):
    """Hemispherical (dipole-modulation) amplitude A, Planck-style idea:
    local variance in caps across the sky, fit variance ~ m(1 + 2A cos)."""
    cosmax = np.cos(np.radians(cap_deg))
    centers = sky_points(192)
    dots = centers @ SKY.T
    local_var = np.array([sky_vals[d > cosmax].var() for d in dots])
    X = np.column_stack([np.ones(len(centers)), centers])
    coef, *_ = np.linalg.lstsq(X, local_var, rcond=None)
    m, d = coef[0], coef[1:]
    A = np.linalg.norm(d) / (2 * m)
    return A, d / np.linalg.norm(d)


LMAX = 20
_theta = np.arccos(np.clip(SKY[:, 2], -1, 1))
_phi = np.arctan2(SKY[:, 1], SKY[:, 0])


def angular_spectrum(sky_vals, lmax=LMAX):
    """Angular power C_l for l = 0..lmax from evenly spread sky points.
    Large l = small patches, small l = giant patches."""
    from scipy.special import sph_harm_y
    w = 4 * np.pi / len(sky_vals)
    cl = np.zeros(lmax + 1)
    for l in range(lmax + 1):
        m = np.arange(-l, l + 1)[:, None]
        Y = sph_harm_y(l, m, _theta[None, :], _phi[None, :])
        alm = w * (Y.conj() @ sky_vals)
        cl[l] = np.mean(np.abs(alm) ** 2)
    return cl


def large_scale_fraction(cl, l_big=(2, 4), l_all=(2, LMAX)):
    """Share of the unevenness (l>=2) that sits in the biggest patches (l=2-4)."""
    v = (2 * np.arange(len(cl)) + 1) * cl
    return v[l_big[0]:l_big[1] + 1].sum() / v[l_all[0]:l_all[1] + 1].sum()


def scale_invariant_reference():
    """What a scale-invariant sky (roughly what the real CMB does on large
    scales: l(l+1)C_l flat) gives for the same measure."""
    l = np.arange(LMAX + 1).astype(float)
    cl = np.zeros(LMAX + 1)
    cl[2:] = 1 / (l[2:] * (l[2:] + 1))
    return large_scale_fraction(cl)


def scenario(flawed, rng):
    A_out, F_out = [], []
    for _ in range(N_RUNS):
        snap_t = run_snap(make_thresholds(rng, flawed), rng)
        sky = read_sky(snap_t)
        A_out.append(asymmetry_A(sky)[0])
        F_out.append(large_scale_fraction(angular_spectrum(sky)))
    return np.array(A_out), np.array(F_out)


def summarise(a):
    return {"median": float(np.median(a)),
            "p16": float(np.percentile(a, 16)),
            "p84": float(np.percentile(a, 84))}


SPEEDS = [64, 8, 2, 1]  # front speeds to pre-register (cells per step)

if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    t0 = time.time()
    F_ref = scale_invariant_reference()
    params = {k: v for k, v in globals().items()
              if k.isupper() and isinstance(v, (int, float))}
    pred = {
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
        "note": "Sandbox look (rule 16). Not a registration. Written before any comparison with real Planck data.",
        "parameters": params,
        "main_measure": "large_scale_fraction = share of unevenness in l=2-4 out of l=2-20",
        "large_scale_fraction_scale_invariant_reference": float(F_ref),
        "by_front_speed": {},
    }
    runs = {}
    A0, F0 = scenario(False, rng)
    pred["perfect_crystal_null"] = {"large_scale_fraction": summarise(F0), "A": summarise(A0)}
    runs["null_F"], runs["null_A"] = F0, A0
    for sp in SPEEDS:
        globals()["FRONT_SPEED"] = sp
        A1, F1 = scenario(True, rng)
        pred["by_front_speed"][str(sp)] = {"large_scale_fraction": summarise(F1), "A": summarise(A1)}
        runs[f"speed{sp}_F"], runs[f"speed{sp}_A"] = F1, A1
    pred["prediction_in_words"] = (
        "A flawed crystal matches a scale-invariant sky only if the snap front is slow "
        "(many flaws go off independently). Fast fronts put far too much unevenness "
        "into the largest patterns; a perfect crystal puts far too little.")
    out = Path(__file__).resolve().parent / "out"
    out.mkdir(exist_ok=True)
    (out / "flaw_grid_snap_look_2026-11-18.json").write_bytes(json.dumps(pred, indent=2).encode())
    np.savez(out / "flaw_grid_snap_look_2026-11-18_runs.npz", **runs)
    print(json.dumps({k: v for k, v in pred.items() if k != "parameters"}, indent=2))
    print(f"done in {time.time() - t0:.0f}s")


def compare_to_planck(path="COM_CMB_IQU-smica_2048_R3.00_full.fits"):
    """STEP 2 (later): measure A on the real Planck map the same way.
    Needs: pip install healpy, and the SMICA map from the Planck Legacy Archive.
    Only after a registration in PREREGISTRATION.md (rule 4)."""
    import healpy as hp
    m = hp.read_map(path)
    m = hp.ud_grade(hp.smoothing(m, fwhm=np.radians(5)), 16)
    vecs = np.array(hp.pix2vec(16, np.arange(hp.nside2npix(16)))).T
    global SKY
    SKY = vecs
    vals = (m - m.mean()) / m.std()
    return asymmetry_A(vals)
