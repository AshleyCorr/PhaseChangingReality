"""Snap test: Ashley's two models for an uneven snap, against the Planck afterglow.

Every setting here comes from DIAL_SHEET.md (locked 11 November 2026, before any
measurement of the real sky). Nothing in this file may be changed after the real-sky run.

Usage
  python snap_test.py check  DATA_DIR OUT_DIR   # fake skies only: does the method recover
                                                # an injected Model A / Model B? (no real data read)
  python snap_test.py run    DATA_DIR OUT_DIR   # the real test, after the code is committed

DATA_DIR holds the files written by prepare_maps.py:
  smica_T_nside128_fwhm1deg.fits   (K_CMB, Galactic, RING)
  common_mask_nside128.fits
"""
import json
import os
import sys
import time

import numpy as np
import healpy as hp

# ---------------------------------------------------------------- locked settings
NSIDE = 128
LMAX = 3 * NSIDE - 1
FWHM_SMOOTH_DEG = 1.0          # the real map was blurred to 1 degree (prepare_maps.py)
SMICA_BEAM_ARCMIN = 5.0        # SMICA's own resolution, applied to the fake skies too
BANDS = {"large": (2, 40), "medium": (41, 80), "small": (81, 160)}
FADE = {"large": 1.0, "medium": 0.5, "small": 0.25}   # A2 and B3: half per smaller band
DISK_RADIUS_DEG = 8.0
DISK_CENTRES_NSIDE = 16        # 3,072 circle centres
DISK_MIN_UNMASKED = 0.9        # a circle counts only if at least 90% of it is unmasked sky
PATCH_FWHM_DEG = 60.0          # B2: 60 degree patches
N_STANDARD = 1000
N_MODEL = 1000
N_CALIB = 100                  # fake skies per trial strength when fitting A1 / B1
CALIB_GRID = np.linspace(0.0, 0.30, 16)
VERDICT = (2.5, 97.5)          # middle 95%
SEED = 20261111
PLANCK2018 = dict(H0=67.32, ombh2=0.02238, omch2=0.1201, mnu=0.06, omk=0,
                  tau=0.0543, As=2.100e-9, ns=0.9660)   # Planck 2018 TT,TE,EE+lowE+lensing best fit


# ---------------------------------------------------------------- set-up
def standard_cl():
    import camb
    p = camb.set_params(lmax=LMAX + 100, **PLANCK2018)
    r = camb.get_results(p)
    cl = r.get_cmb_power_spectra(p, CMB_unit="muK", raw_cl=True)["total"][: LMAX + 1, 0]
    cl[:2] = 0.0
    beam = hp.gauss_beam(np.radians(SMICA_BEAM_ARCMIN / 60), lmax=LMAX) * \
        hp.gauss_beam(np.radians(FWHM_SMOOTH_DEG), lmax=LMAX)
    return cl * beam ** 2


def band_filters():
    out = {}
    for name, (lo, hi) in BANDS.items():
        f = np.zeros(LMAX + 1)
        f[lo:hi + 1] = 1.0
        out[name] = f
    return out


def disks(mask):
    """Pixel lists for the circles that are at least 90% unmasked, and their centre vectors."""
    keep, centres = [], []
    for c in range(hp.nside2npix(DISK_CENTRES_NSIDE)):
        v = hp.pix2vec(DISK_CENTRES_NSIDE, c)
        pix = hp.query_disc(NSIDE, v, np.radians(DISK_RADIUS_DEG))
        good = pix[mask[pix] > 0]
        if len(good) >= DISK_MIN_UNMASKED * len(pix):
            keep.append(good)
            centres.append(v)
    return keep, np.array(centres)


G = {}   # shared state for worker processes


def init(cl, mask_path):
    mask = hp.read_map(mask_path, dtype=np.float64)
    G["mask"] = mask
    G["cl"] = cl
    G["filters"] = band_filters()
    G["disks"], G["centres"] = disks(mask)
    G["pixvec"] = np.array(hp.pix2vec(NSIDE, np.arange(hp.nside2npix(NSIDE))))
    G["design"] = np.column_stack([np.ones(len(G["centres"])), G["centres"]])


# ---------------------------------------------------------------- measuring
def band_maps(full_sky_map):
    """Zero the masked sky, then split into the three ripple bands."""
    m = full_sky_map * G["mask"]
    alm = hp.map2alm(m, lmax=LMAX, iter=1)
    return {b: hp.alm2map(hp.almxfl(alm, f), NSIDE, lmax=LMAX) for b, f in G["filters"].items()}


def local_power(m):
    return np.array([m[p].var() for p in G["disks"]])


def measure(local, expected):
    """Lopsidedness, direction and patchiness of one band's circle-by-circle ripple strength."""
    delta = local / expected - 1.0
    coef, *_ = np.linalg.lstsq(G["design"], delta, rcond=None)
    dip = coef[1:]
    resid = delta - G["design"] @ coef
    amp = np.linalg.norm(dip)
    return {"A": amp / 2.0,                       # power tilts as 2A for ripples tilted by A
            "dir": (dip / amp).tolist() if amp > 0 else [0.0, 0.0, 1.0],
            "patch": float(resid.std())}


# ---------------------------------------------------------------- fake skies
def iso_sky(rng):
    alm = hp.synalm(G["cl"], lmax=LMAX, new=True)
    return hp.alm2map(alm, NSIDE, lmax=LMAX)


def split_full(m):
    alm = hp.map2alm(m, lmax=LMAX, iter=1)
    return {b: hp.alm2map(hp.almxfl(alm, f), NSIDE, lmax=LMAX) for b, f in G["filters"].items()}


def modulated_sky(rng, kind, strength, direction=None):
    """Model A: 1 + A_b cos(angle to direction). Model B: 1 + B_b g, g = 60 degree patches.
    The modulation is applied band by band, with the locked half-per-band fade."""
    m = iso_sky(rng)
    if kind == "standard" or strength == 0:
        return m
    parts = split_full(m)
    if kind == "A":
        shape = np.asarray(direction) @ G["pixvec"]
    else:
        g = hp.smoothing(rng.standard_normal(hp.nside2npix(NSIDE)), fwhm=np.radians(PATCH_FWHM_DEG))
        g = g - g.mean()
        shape = g / g.std()
    out = np.zeros_like(m)
    for b, part in parts.items():
        out += part * (1.0 + strength * FADE[b] * shape)
    return out


def one_sky(args):
    seed, kind, strength, direction = args
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    bm = band_maps(modulated_sky(rng, kind, strength, direction))
    return {b: local_power(bm[b]) for b in BANDS}


CACHE = {"dir": None}


def many(pool, n, kind, strength, direction, seed0):
    """Make n fake skies. Each batch is saved, so an interrupted run picks up where it stopped
    (same seeds, so the same skies)."""
    skies = []
    for c0 in range(0, n, 100):
        c1 = min(n, c0 + 100)
        key = f"{kind}_{strength:.4f}_{seed0}_{c0}_{c1}"
        if direction is not None:
            key += "_" + "_".join(f"{x:.4f}" for x in direction)
        path = os.path.join(CACHE["dir"], key + ".npz") if CACHE["dir"] else None
        if path and os.path.exists(path):
            z = np.load(path, allow_pickle=True)
            skies += list(z["skies"])
            continue
        chunk = [one_sky((seed0 + i, kind, strength, direction)) for i in range(c0, c1)]
        if path:
            np.savez_compressed(path, skies=np.array(chunk, dtype=object))
        skies += chunk
    return skies


# ---------------------------------------------------------------- the analysis
def summarise(locals_, expected):
    return {b: measure(locals_[b], expected[b]) for b in BANDS}


def angle_deg(u, v):
    return float(np.degrees(np.arccos(np.clip(np.dot(u, v), -1, 1))))


def stats_table(skies, expected):
    """Per band: arrays of A and patchiness, plus band-to-band direction angles."""
    rows = [summarise(s, expected) for s in skies]
    out = {}
    for b in BANDS:
        out[b] = {"A": np.array([r[b]["A"] for r in rows]),
                  "patch": np.array([r[b]["patch"] for r in rows])}
    out["angle_large_medium"] = np.array([angle_deg(r["large"]["dir"], r["medium"]["dir"]) for r in rows])
    out["angle_large_small"] = np.array([angle_deg(r["large"]["dir"], r["small"]["dir"]) for r in rows])
    return out


def place(value, sample):
    lo, hi = np.percentile(sample, VERDICT)
    pct = float((sample < value).mean() * 100)
    return {"value": float(value), "percentile": pct, "middle95": [float(lo), float(hi)],
            "inside": bool(lo <= value <= hi), "median": float(np.median(sample))}


def calibrate(pool, kind, target_stat, target_value, direction, expected, seed0):
    """Pick the strength whose fake skies have a median large-band statistic closest to the real sky's."""
    best, rows = None, []
    for k, s in enumerate(CALIB_GRID):
        skies = many(pool, N_CALIB, kind, s, direction, seed0 + 10_000 * k)
        med = float(np.median(stats_table(skies, expected)["large"][target_stat]))
        rows.append([float(s), med])
        if best is None or abs(med - target_value) < abs(best[1] - target_value):
            best = (float(s), med)
    return best[0], rows


def real_or_injected(mode, data, pool, expected, seed0):
    if mode == "run":
        t = hp.read_map(os.path.join(data, "smica_T_nside128_fwhm1deg.fits"), dtype=np.float64) * 1e6
        bm = band_maps(t)
        return {b: local_power(bm[b]) for b in BANDS}, "Planck 2018 SMICA"
    # check mode: a fake sky with a known Model A injected (strength 0.15, a fixed direction)
    inj_dir = hp.ang2vec(224.0, -22.0, lonlat=True)
    return many(pool, 1, "A", 0.15, inj_dir, seed0)[0], "injected Model A, strength 0.15"


def main(mode, data, out):
    os.makedirs(out, exist_ok=True)
    CACHE["dir"] = os.path.join(out, "cache")
    os.makedirs(CACHE["dir"], exist_ok=True)
    t0 = time.time()
    cl = standard_cl()
    mask_path = os.path.join(data, "common_mask_nside128.fits")
    init(cl, mask_path)
    report = {"mode": mode, "settings_from": "DIAL_SHEET.md (locked 11 November 2026)",
              "n_disks": len(G["disks"]), "sky_fraction": float(G["mask"].mean())}
    pool = None   # run in one process (worker pools hang on this machine); batches are cached
    if True:
        std = many(pool, N_STANDARD, "standard", 0.0, None, SEED)
        expected = {b: np.mean([s[b] for s in std], axis=0) for b in BANDS}
        std_stats = stats_table(std, expected)

        target_locals, label = real_or_injected(mode, data, pool, expected, SEED + 9_000_000)
        target = summarise(target_locals, expected)
        report["target"] = label
        report["target_measured"] = target
        ang_lm = angle_deg(target["large"]["dir"], target["medium"]["dir"])
        ang_ls = angle_deg(target["large"]["dir"], target["small"]["dir"])

        def verdicts(sample_stats):
            v = {}
            for b in BANDS:
                v[b] = {"lopsidedness": place(target[b]["A"], sample_stats[b]["A"]),
                        "patchiness": place(target[b]["patch"], sample_stats[b]["patch"])}
            v["direction_large_vs_medium_deg"] = place(ang_lm, sample_stats["angle_large_medium"])
            v["direction_large_vs_small_deg"] = place(ang_ls, sample_stats["angle_large_small"])
            return v

        report["standard"] = verdicts(std_stats)

        direction = np.array(target["large"]["dir"])
        a1, a_rows = calibrate(pool, "A", "A", target["large"]["A"], direction, expected, SEED + 1_000_000)
        model_a = stats_table(many(pool, N_MODEL, "A", a1, direction, SEED + 2_000_000), expected)
        report["model_A"] = {"A1_fitted_on_large_band": a1, "calibration": a_rows, "verdicts": verdicts(model_a)}

        total_large = float(np.sqrt(target["large"]["patch"] ** 2 + (2 * target["large"]["A"]) ** 2))
        def total_unevenness(stats):
            return np.sqrt(stats["large"]["patch"] ** 2 + (2 * stats["large"]["A"]) ** 2)
        best, b_rows = None, []
        for k, s in enumerate(CALIB_GRID):
            sk = stats_table(many(pool, N_CALIB, "B", s, None, SEED + 3_000_000 + 10_000 * k), expected)
            med = float(np.median(total_unevenness(sk)))
            b_rows.append([float(s), med])
            if best is None or abs(med - total_large) < abs(best[1] - total_large):
                best = (float(s), med)
        b1 = best[0]
        model_b = stats_table(many(pool, N_MODEL, "B", b1, None, SEED + 4_000_000), expected)
        report["model_B"] = {"B1_fitted_on_large_band": b1, "calibration": b_rows, "verdicts": verdicts(model_b)}

    report["minutes"] = round((time.time() - t0) / 60, 1)
    with open(os.path.join(out, f"snap_test_{mode}.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps({k: report[k] for k in ("mode", "target", "minutes")}))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
