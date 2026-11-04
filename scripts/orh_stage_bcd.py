"""Stages B, C and D of the observer-relative-histories handoff on the unitary toy, written to results/ and
docs/figures/. Usage:

    python scripts/orh_stage_bcd.py configs/orh_stage_bcd.json [out_dir]

Every row is exact (a closed form or an exact state-vector computation checked by tests/test_orh_*.py); no random
number is drawn. One CSV, `results/orh_stage_bcd.csv`, holds every row with a `block` column saying which table it
belongs to; the figures are the Stage F items (1) to (4) of the handoff:

    orh_stage_b_visibility.png            visibility against the environment overlap, and against the record angle
    orh_stage_b_partial_information.png   I(S:F) against fragment size for several record strengths
    orh_stage_c_decoherence_functional.png  |D(alpha, beta)| for the two-slit family at four record strengths
    orh_stage_c_nested_measure.png        the measure of one record's compatible set falling while the total stays one
    orh_stage_d_joint_records.png         the joint distribution of two observers' records against the product
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphity.results import ResultWriter, provenance        # noqa: E402
from orh import __version__                                   # noqa: E402
from orh import darwinism as dw                               # noqa: E402
from orh import histories as hist                             # noqa: E402
from orh.observers import World, disagreement                 # noqa: E402
from orh.statevector import Register, binary_entropy, visibility  # noqa: E402

# the validated default palette of the dataviz skill, categorical slots in fixed order; text and surface tokens
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, SURFACE, RULE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"


def style(ax, title, xlabel, ylabel):
    ax.set_facecolor(SURFACE)
    ax.set_title(title, color=INK, loc="left", fontsize=11)
    ax.set_xlabel(xlabel, color=INK2)
    ax.set_ylabel(ylabel, color=INK2)
    ax.grid(True, color=RULE, linewidth=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(RULE)
    ax.tick_params(colors=INK2)


def stage_b(cfg, out, figdir, plt):
    b = cfg["stage_b"]
    rows = []
    # (1) visibility against the environment overlap: one environment qubit, the angle scanned
    angles = np.deg2rad(b["record_angles_deg"])
    for th in angles:
        reg = Register(["s", "e"]).prepare("s", b["p_one"]).controlled_tilt("s", "e", th)
        v_sim = visibility(reg.reduced(["s"]))
        rows.append(dict(block="B_visibility", angle_deg=np.rad2deg(th), overlap=np.cos(th), p_one=b["p_one"],
                         visibility_sim=v_sim, visibility_exact=dw.visibility(b["p_one"], [th]),
                         environment_entropy_bits=reg.entropy(["e"])))
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), facecolor=SURFACE)
    ov = [r["overlap"] for r in rows if r["block"] == "B_visibility"]
    vs = [r["visibility_sim"] for r in rows if r["block"] == "B_visibility"]
    axes[0].plot(ov, vs, color=SERIES[0], linewidth=2, marker="o", markersize=6)
    style(axes[0], "Visibility equals the environment overlap", "overlap  |<E_B|E_A>|", "fringe visibility V")
    axes[1].plot([r["angle_deg"] for r in rows], vs, color=SERIES[0], linewidth=2, marker="o", markersize=6)
    style(axes[1], "Against the record angle (90 deg = a perfect copy)", "record angle (degrees)", "V")
    fig.tight_layout()
    fig.savefig(figdir / "orh_stage_b_visibility.png", dpi=160)
    plt.close(fig)

    # (2) the partial-information plot and redundancy
    pi = b["partial_information"]
    fig, ax = plt.subplots(figsize=(6.4, 4), facecolor=SURFACE)
    for k, deg in enumerate(pi["angles_deg"]):
        th = np.deg2rad(deg)
        plot = dw.partial_information_plot(pi["p_one"], th, pi["n_env"])
        # the simulator at every fragment size, as the exact check written into the table
        reg = Register(["s"] + [f"e{i}" for i in range(pi["n_env"])]).prepare("s", pi["p_one"])
        for i in range(pi["n_env"]):
            reg.controlled_tilt("s", f"e{i}", th)
        for m in range(pi["n_env"] + 1):
            sim = reg.mutual_information(["s"], [f"e{i}" for i in range(m)]) if m else 0.0
            rows.append(dict(block="B_partial_information", angle_deg=deg, p_one=pi["p_one"], n_env=pi["n_env"],
                             fragment_size=m, I_SF_exact=float(plot[m]), I_SF_sim=sim,
                             H_S_bits=binary_entropy(pi["p_one"])))
        ax.plot(range(pi["n_env"] + 1), plot, color=SERIES[k], linewidth=2, marker="o", markersize=6,
                label=f"record angle {deg} deg")
        ax.annotate(f"{deg} deg", (pi["n_env"], plot[-1]), textcoords="offset points", xytext=(6, 6 - 9 * k),
                    color=INK2, fontsize=8)
    ax.axhline(binary_entropy(pi["p_one"]), color=INK2, linewidth=1, linestyle=":")
    ax.text(0.2, binary_entropy(pi["p_one"]) + 0.03, "H(S): the classical plateau", color=INK2, fontsize=8)
    style(ax, "Partial information I(S:F) against fragment size", "fragment size (environment qubits)",
          "I(S:F) in bits")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(figdir / "orh_stage_b_partial_information.png", dpi=160)
    plt.close(fig)
    rd = b["redundancy"]
    for deg in rd["angles_deg"]:
        rows.append(dict(block="B_redundancy", angle_deg=deg, p_one=rd["p_one"], n_env=rd["n_env"],
                         delta=rd["delta"], redundancy=dw.redundancy(rd["p_one"], np.deg2rad(deg), rd["n_env"],
                                                                     rd["delta"])))
    # (3) eraser and recombination, exact rows
    reg = Register(["s", "e"]).h("s").cnot("s", "e")
    v_recorded = visibility(reg.reduced(["s"]))
    reg.h("e")
    v_sorted = [visibility(reg.projected({"e": v}).reduced(["s"])) for v in (0, 1)]
    v_unsorted = visibility(reg.reduced(["s"]))
    rows.append(dict(block="B_eraser", visibility_after_record=v_recorded, visibility_sorted_e0=v_sorted[0],
                     visibility_sorted_e1=v_sorted[1], visibility_unsorted=v_unsorted))
    reg = Register(["s", "e"]).h("s").controlled_tilt("s", "e", 0.8)
    v_before = visibility(reg.reduced(["s"]))
    reg.controlled_tilt("s", "e", -0.8)
    rows.append(dict(block="B_recombination", angle_rad=0.8, visibility_recorded=v_before,
                     visibility_after_inverse=visibility(reg.reduced(["s"]))))
    for r in rows:
        out.write(r)


def stage_c(cfg, out, figdir, plt):
    c = cfg["stage_c"]
    angles = c["two_slit_record_angles_deg"]
    fig, axes = plt.subplots(1, len(angles), figsize=(2.6 * len(angles), 3.0), facecolor=SURFACE)
    for ax, deg in zip(axes, angles):
        th = np.deg2rad(deg)
        steps = [hist.step(lambda r, th=th: r.h("s").controlled_tilt("s", "e", th), hist.alternatives_over(["s"]),
                           "path"),
                 hist.step(lambda r: r.h("s"), hist.alternatives_over(["s"]), "output")]
        labels, d = hist.decoherence_functional(Register(["s", "e"]), steps)
        classes = {"out0": [(0, 0), (1, 0)], "out1": [(0, 1), (1, 1)]}
        viol = hist.sum_rule_violations(labels, d, classes)
        for a, la in enumerate(labels):
            for b_, lb in enumerate(labels):
                out.write(dict(block="C_decoherence_functional", angle_deg=deg, alpha=str(la), beta=str(lb),
                               D_real=float(d[a, b_].real), D_imag=float(d[a, b_].imag), D_abs=float(abs(d[a, b_]))))
        out.write(dict(block="C_sum_rules", angle_deg=deg, total_weight=hist.total_weight(d),
                       max_offdiagonal=hist.max_offdiagonal(d), decoherent=hist.is_decoherent(d),
                       out0_fine=viol["out0"]["fine"], out0_coarse=viol["out0"]["coarse"],
                       out1_fine=viol["out1"]["fine"], out1_coarse=viol["out1"]["coarse"]))
        im = ax.imshow(np.abs(d), cmap="Blues", vmin=0, vmax=0.25)
        ax.set_xticks(range(4)); ax.set_yticks(range(4))
        ax.set_xticklabels([f"{l[0]}{l[1]}" for l in labels], fontsize=7, color=INK2)
        ax.set_yticklabels([f"{l[0]}{l[1]}" for l in labels], fontsize=7, color=INK2)
        ax.set_title(f"record angle {deg} deg", fontsize=9, color=INK, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle("|D(alpha, beta)| of the two-slit family (path, output); off-diagonal terms shrink with the record",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02).set_label("|D|", color=INK2)
    fig.savefig(figdir / "orh_stage_c_decoherence_functional.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    # nested records: n systems, each recorded by its own register
    nr = c["nested_records"]
    ps = nr["p_ones"][: nr["n_steps"]]
    names = [f"s{i}" for i in range(len(ps))] + [f"o{i}" for i in range(len(ps))]
    steps = [hist.step(lambda r, i=i, p=p: r.prepare(f"s{i}", p).cnot(f"s{i}", f"o{i}"),
                       hist.alternatives_over([f"o{i}"]), f"record {i}") for i, p in enumerate(ps)]
    labels, d = hist.decoherence_functional(Register(names), steps)
    # the record sequence: value 1 at every step (the chosen outcome; any sequence nests the same way)
    records = [{k: 1 for k in range(n)} for n in range(1, len(ps) + 1)]
    rows = hist.nested_record_table(labels, d, records)
    for row in rows:
        out.write(dict(block="C_nested_measure", n_records=row["n"], compatible_size=row["size"], mu=row["mu"],
                       surprisal_bits=row["surprisal_bits"], conditional_total=row["conditional_total"],
                       nested=row["nested"], family_decoherent=row["family_decoherent"],
                       total_weight=row["total_weight"], mu_exact=float(np.prod(ps[: row["n"]]))))
    fig, ax = plt.subplots(figsize=(6.4, 3.8), facecolor=SURFACE)
    ns = [r["n"] for r in rows]
    ax.plot(ns, [r["total_weight"] for r in rows], color=SERIES[1], linewidth=2, marker="o", markersize=6,
            label="total weight of all histories")
    ax.plot(ns, [r["mu"] for r in rows], color=SERIES[0], linewidth=2, marker="o", markersize=6,
            label="measure of the histories compatible with the record")
    ax.plot(ns, [r["conditional_total"] for r in rows], color=SERIES[2], linewidth=2, linestyle="--",
            label="conditional measure inside the compatible set")
    ax.set_xticks(ns)
    style(ax, "One record's compatible set shrinks; the total stays one", "records retained",
          "Born measure")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "orh_stage_c_nested_measure.png", dpi=160)
    plt.close(fig)

    # the erased record: the family that includes it is not decoherent
    steps = [hist.step(lambda r: r.h("s").cnot("s", "e"), hist.alternatives_over(["e"]), "record"),
             hist.step(lambda r: r.cnot("s", "e").h("s"), hist.alternatives_over(["s"]), "output after erasure")]
    labels, d = hist.decoherence_functional(Register(["s", "e"]), steps)
    out.write(dict(block="C_erased_record", max_offdiagonal=hist.max_offdiagonal(d), decoherent=hist.is_decoherent(d),
                   total_weight=hist.total_weight(d)))


def stage_d(cfg, out, figdir, plt):
    dcfg = cfg["stage_d"]
    p = dcfg["p_one"]
    joints = []
    for ti_deg, tj_deg in dcfg["fragment_angles_deg"]:
        ti, tj = np.deg2rad(ti_deg), np.deg2rad(tj_deg)
        w = World(["s", "e0", "e1", "oa", "ob"]).prepare("s", p).tilt_record("s", "e0", ti).tilt_record("s", "e1", tj)
        w.observe("A", "e0", "oa").observe("B", "e1", "ob")
        joint = w.joint({"A": ["oa"], "B": ["ob"]})
        prod = w.product_of_marginals({"A": ["oa"], "B": ["ob"]})
        for key in sorted(set(joint) | set(prod)):
            out.write(dict(block="D_joint_records", angle_a_deg=ti_deg, angle_b_deg=tj_deg, p_one=p,
                           record_a=key[0][0], record_b=key[1][0], joint=joint.get(key, 0.0),
                           product_of_marginals=prod.get(key, 0.0)))
        out.write(dict(block="D_disagreement", angle_a_deg=ti_deg, angle_b_deg=tj_deg, p_one=p,
                       disagreement_sim=disagreement(joint), disagreement_exact=dw.disagreement_rate(p, ti, tj),
                       record_mutual_information_bits=w.record_mutual_information(["oa"], ["ob"])))
        joints.append((ti_deg, tj_deg, joint, prod))
    if dcfg.get("told_record"):
        w = World(["s", "e", "oa", "ob"]).prepare("s", p).tilt_record("s", "e", np.pi / 2).observe("A", "e", "oa")
        before = w.record_mutual_information(["oa"], ["ob"])
        w.tell("A", "oa", "B", "ob")
        out.write(dict(block="D_told_record", p_one=p, mutual_information_before=before,
                       mutual_information_after=w.record_mutual_information(["oa"], ["ob"]),
                       disagreement_after=disagreement(w.joint({"A": ["oa"], "B": ["ob"]}))))
    fig, axes = plt.subplots(1, len(joints), figsize=(2.9 * len(joints), 3.4), facecolor=SURFACE, sharey=True)
    keys = [((0,), (0,)), ((0,), (1,)), ((1,), (0,)), ((1,), (1,))]
    x = np.arange(4)
    for ax, (ti_deg, tj_deg, joint, prod) in zip(axes, joints):
        ax.bar(x - 0.2, [joint.get(k, 0.0) for k in keys], width=0.38, color=SERIES[0], label="joint (Born)")
        ax.bar(x + 0.2, [prod.get(k, 0.0) for k in keys], width=0.38, color=SERIES[1],
               label="product of marginals")
        ax.set_xticks(x)
        ax.set_xticklabels(["A0 B0", "A0 B1", "A1 B0", "A1 B1"], fontsize=8, color=INK2)
        style(ax, f"records of {ti_deg} and {tj_deg} deg", "", "probability" if ax is axes[0] else "")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Two observers' records: the joint distribution against the product of their marginals",
                 fontsize=9, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(figdir / "orh_stage_d_joint_records.png", dpi=160)
    plt.close(fig)


def main(argv):
    cfg_path = (Path(argv[1]) if len(argv) > 1 else ROOT / "configs" / "orh_stage_bcd.json").resolve()
    out_dir = Path(argv[2]) if len(argv) > 2 else ROOT / "results"
    cfg = json.loads(cfg_path.read_text())
    figdir = ROOT / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    meta = dict(config=cfg, config_path=str(cfg_path.relative_to(ROOT)), orh_version=__version__,
                python=platform.python_version(), numpy=np.__version__, exact=True, **provenance(ROOT))
    rows = []

    class Collector:
        def write(self, row):
            rows.append(dict(row))

    collector = Collector()
    stage_b(cfg, collector, figdir, plt)
    stage_c(cfg, collector, figdir, plt)
    stage_d(cfg, collector, figdir, plt)
    # the blocks have different columns; every row gets the union, in order of first appearance, blanks elsewhere
    columns = []
    for r in rows:
        for k in r:
            if k not in columns:
                columns.append(k)
    with ResultWriter(cfg["name"], meta, out_dir=str(out_dir)) as out:
        for r in rows:
            out.write({k: r.get(k, "") for k in columns})
    print(f"wrote {out_dir / (cfg['name'] + '.csv')} and figures in {figdir}")


if __name__ == "__main__":
    main(sys.argv)
