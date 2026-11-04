"""Write the configs of T74's registered stage 1 (PREREGISTRATION.md T74, "Design (fixed now)" as repaired by its
Amendments 2 and 3; ASSUMPTIONS Q24). Usage:

    python scripts/make_t74_configs.py [--out configs] [--held-out]

108 cells, one config each: lambda 1.10, 1.25, 1.40 x L 8, 12 x feed steady (R every sweep), small meals (16 R every
16 sweeps), large meals (256 R every 256 sweeps) x cap c = 2R, R/2 x unfed background cold (g_bg = 0), equal (the
common mean of reservoir and slab stores when the unfed phase starts, read from the run; Amendment 3), warm (4a);
R = r N_slab, r = 1/16; n_out = N_slab; gate g1 = a N_slab; T71's model arm (layer moves plus the clean two-switch move,
max width 8, one layer proposal a sweep); every release to the daughter ledger; fed for 4,096 sweeps with the reservoir
isolated (rho = 0); then digestion, still isolated, until the reservoir's and the slab's mean store energies agree
within 1 % or 4,096 sweeps pass (Amendment 3); then unfed for 4,096 sweeps relaxing at 1/64; 16 replicas; seeds 20267401 to 20267508, one per cell, in the
order lambda, L, feed, cap, background (the loops below, outermost first). The held-out L = 16 is NOT generated: it
runs only after L = 8 and 12 are read and its numbers are written (rule 14).

`--held-out` (added 2026-10-26, PREREGISTRATION T74 Amendment 6) writes the 54 L = 16 configs instead: every key as in
stage 1 except L = 16, the seed (20267901 to 20267954, one per cell, in the order lambda, feed, cap, background), the
name, `written` and `_purpose`. It refuses to write until the prediction file configs/t74_heldout_prediction.json exists.

The protocol's choices that are ours (ASSUMPTIONS Q24): the reservoir starts empty (e_out0 = 0); a meal arrives at the
start of its period (sweeps 1, 1 + T, ...), so over 4,096 fed sweeps every feed delivers 4,096 R; a store is never
taken below zero; the stored energy read is the sum of the slab's stores.

An existing config with different content is never overwritten; an identical one is left as it is.
"""
import json
import sys
from pathlib import Path

LAMBDAS = (1.10, 1.25, 1.40)
SIZES = (8, 12)
FEEDS = (("steady", "steady", 1), ("meals16", "meals", 16), ("meals256", "meals", 256))
CAPS = (("c2R", "2 * R"), ("cRhalf", "R / 2"))
BACKGROUNDS = (("cold", "0"), ("equal", "equal"), ("warm", "4 * a"))
SEED0 = 20267401
WRITTEN = "2026-10-20"
HELD_OUT_L = 16
HELD_OUT_SEED0 = 20267901
HELD_OUT_WRITTEN = "2026-10-26"
PREDICTION = Path("configs") / "t74_heldout_prediction.json"

PURPOSE = ("PREREGISTRATION.md T74 as repaired by its Amendments 2 and 3, registered stage 1 (L = 8 and 12; the "
           "held-out L = 16 is not generated; VISION Updates 72 and 73; ASSUMPTIONS Q24): the owner's slab 4 x L x L "
           "under the layer form, T71's model arm, fed through a reservoir and a curl edge with one cap c both ways; the "
           "reservoir isolated for the 4,096 fed sweeps and through digestion (until the two mean store energies agree "
           "within 1 % or 4,096 sweeps pass), then relaxing at 1/64 for 4,096 sweeps toward the unfed background (cold; "
           "equal, the common mean of reservoir and slab at that moment; or warm). Not exploratory. Every six-link result carries VISION Update 24's caveat. Not to be launched "
           "before the owner confirms or revises her prediction against Amendment 2, item 5.")


PURPOSE_HELD_OUT = ("PREREGISTRATION.md T74, Amendment 6 (2026-10-26): the registered held-out size L = 16 (rule 14), "
                    "run after L = 8 and 12 were read (O145) and its numbers written from them alone "
                    "(configs/t74_heldout_prediction.json); every other key as in stage 1. The owner's word of "
                    "2026-10-25 (VISION Update 115): the size claim confirmed and the runs go ahead. Not exploratory. "
                    "Every six-link result carries VISION Update 24's caveat, and Update 95's statement: the model's "
                    "author advised against the six-link version, and the field's prior is a two-dimensional base "
                    "with the third dimension emergent.")


def held_out_cells():
    i = 0
    for cfg in cells(sizes=(HELD_OUT_L,)):
        cfg.update(_purpose=PURPOSE_HELD_OUT, written=HELD_OUT_WRITTEN, seed=HELD_OUT_SEED0 + i)
        i += 1
        yield cfg


def cells(sizes=SIZES):
    i = 0
    for lam in LAMBDAS:
        for L in sizes:
            for feed_label, feed, every in FEEDS:
                for cap_label, c in CAPS:
                    for bg_label, g_bg in BACKGROUNDS:
                        name = "t74_lam%d_L%02d_%s_%s_%s" % (round(lam * 100), L, feed_label, cap_label, bg_label)
                        yield dict(
                            _purpose=PURPOSE, name=name, section="T74", written=WRITTEN, lam=lam, L=L, r=1.0 / 16,
                            feed=feed, feed_label=feed_label, every=every, c=c, cap_label=cap_label,
                            n_out="N_slab", background=bg_label, rho_fed=0.0, g_bg_fed="0", rho_unfed=1.0 / 64,
                            g_bg_unfed=g_bg, n_fed="4096", n_digest_max="4096", digest_tol=0.01, n_unfed="4096",
                            gate="a * N_slab",
                            e_out0=0.0, move_set="model", release_to="daughter", max_width=8, log_every=50,
                            log_events=True, replicas=16, seed=SEED0 + i)
                        i += 1


def main(out="configs", held_out=False):
    out = Path(out)
    if held_out and not PREDICTION.exists():
        raise FileNotFoundError("%s does not exist; the held-out configs are written only after it" % PREDICTION)
    written = same = 0
    for cfg in (held_out_cells() if held_out else cells()):
        p = out / (cfg["name"] + ".json")
        text = json.dumps(cfg, indent=2) + "\n"
        if p.exists():
            if p.read_text() != text:
                raise FileExistsError("%s exists with different content; not overwritten" % p)
            same += 1
            continue
        p.write_text(text)
        written += 1
    print("wrote %d configs, %d already present and identical" % (written, same))


if __name__ == "__main__":
    args = sys.argv[1:]
    main(args[args.index("--out") + 1] if "--out" in args else "configs", held_out="--held-out" in args)
