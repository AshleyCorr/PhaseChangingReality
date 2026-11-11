"""Shrink the Planck SMICA temperature map and the common mask to a low-detail
working size for the large-scale snap test. This only reshapes the data; it does
NOT measure anything on the real sky (that waits until the dials are locked)."""
import sys, numpy as np, healpy as hp
src, mask_src, out = sys.argv[1], sys.argv[2], sys.argv[3]
NSIDE, LMAX, FWHM_DEG = 128, 3 * 128 - 1, 1.0
T = hp.read_map(src, field=0, nest=False, dtype=np.float64)       # I_STOKES, K_CMB, Galactic
alm = hp.map2alm(T, lmax=LMAX, iter=1)
alm = hp.smoothalm(alm, fwhm=np.radians(FWHM_DEG), inplace=False)
T_lo = hp.alm2map(alm, NSIDE, lmax=LMAX)
M = hp.read_map(mask_src, nest=False, dtype=np.float64)
M_lo = (hp.ud_grade(M, NSIDE) > 0.9).astype(np.int8)
hp.write_map(f"{out}/smica_T_nside{NSIDE}_fwhm{FWHM_DEG:g}deg.fits", T_lo, overwrite=True, dtype=np.float64)
hp.write_map(f"{out}/common_mask_nside{NSIDE}.fits", M_lo, overwrite=True, dtype=np.int16)
print("done: nside", NSIDE, "sky kept by mask: %.1f%%" % (100 * M_lo.mean()))
