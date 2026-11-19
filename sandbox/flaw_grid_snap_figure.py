"""Sandbox look (18 November 2026): the figure for flaw_grid_snap_look.py. Run that first; reads and writes sandbox/out/."""
import sys
from pathlib import Path
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
OUT = Path(__file__).resolve().parent / "out"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import flaw_grid_snap_look as F

runs = np.load(OUT / "flaw_grid_snap_look_2026-11-18_runs.npz")
ref = F.scale_invariant_reference()
rng = np.random.default_rng(7)
mid = F.N // 2

examples = []
for label, flawed, sp in [("Fast front (speed 64)", True, 64),
                          ("Slow front (speed 1)", True, 1),
                          ("Perfect crystal", False, 1)]:
    F.FRONT_SPEED = sp
    st = F.run_snap(F.make_thresholds(rng, flawed), rng)
    sky = F.read_sky(st)
    examples.append((label, st, sky, F.large_scale_fraction(F.angular_spectrum(sky))))

fig = plt.figure(figsize=(14, 10.5))
lon = np.arctan2(F.SKY[:, 1], F.SKY[:, 0]); lat = np.arcsin(F.SKY[:, 2])
for i, (label, st, sky, frac) in enumerate(examples):
    a = fig.add_subplot(3, 3, 1 + i)
    a.imshow(st[:, :, mid], cmap="magma"); a.set_title(f"{label}\nwhen each cell snapped (dark = early)", fontsize=10)
    a.set_xticks([]); a.set_yticks([])
    b = fig.add_subplot(3, 3, 4 + i, projection="mollweide")
    b.scatter(lon, lat, c=sky, s=5, cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    b.set_title(f"'Sky' — big-patch share = {frac:.2f}", fontsize=10); b.set_xticklabels([]); b.set_yticklabels([])

labels = ["perfect"] + [f"speed {s}" for s in F.SPEEDS]
F_data = [runs["null_F"]] + [runs[f"speed{s}_F"] for s in F.SPEEDS]
A_data = [runs["null_A"]] + [runs[f"speed{s}_A"] for s in F.SPEEDS]

c = fig.add_subplot(3, 2, 5)
c.boxplot(F_data, tick_labels=labels)
c.axhline(ref, color="tab:green", ls="--", label=f"scale-invariant sky ≈ {ref:.2f}")
c.set_ylabel("Share of unevenness in biggest patches (l=2–4)")
c.set_title("Main measure: big vs small patches", fontsize=11); c.legend(fontsize=9)

d = fig.add_subplot(3, 2, 6)
d.boxplot(A_data, tick_labels=labels)
d.axhline(F.PLANCK_A_REFERENCE, color="tab:red", ls="--", label="Planck ≈ 0.07 (to verify)")
d.set_ylabel("Lopsidedness A"); d.set_title("Secondary measure: lopsidedness", fontsize=11); d.legend(fontsize=9)

fig.suptitle("Flaw-grid snap: a slow front matches the real sky's big-vs-small mix", fontsize=14)
fig.tight_layout(); fig.savefig(OUT / "flaw_grid_snap_look_2026-11-18.png", dpi=130)
print("saved")
