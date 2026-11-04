"""Stage E of the observer-relative-histories handoff: the observer-moment rate over cosmic time, per comoving volume,
in one fixed Planck-2018 background, under the scenario grid of the config. Usage:

    python scripts/orh_stage_e.py configs/orh_stage_e.json [out_dir]
    python scripts/orh_stage_e.py configs/orh_stage_e.json [results_dir] --figures-only

The first form refuses to run if its result files already exist (graphity.results.ResultWriter; results are
append-only). The second redraws figure 6 from the summary CSV already in results/ and writes nothing there.

Writes two result files and two figures:
    results/orh_stage_e_summary.csv   one row per (scenario, reference class, interaction gamma, exotic eps):
                                      F(13.8 Gyr), the peak of dP/d ln t, the median time, the 99.99 % time, the
                                      last supporting time, whether the integral converged
    results/orh_stage_e_curves.csv    dP/d ln t and F(t) on a coarse log grid for every scenario at gamma = 0,
                                      eps = 0, every reference class
    docs/figures/orh_stage_e_w_of_t.png      dP/d ln t for the main scenarios (log time axis)
    docs/figures/orh_stage_e_f_today.png     F(13.8 Gyr) by scenario and reference class

All inputs are in src/orh/cosmic.py with their sources; the knobs are synthetic and say so. F(13.8 Gyr) is a
description under a declared sampling rule, never evidence for one (Hartle and Srednicki 2007, Eq. 4.2).
"""
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from graphity.results import ResultWriter, provenance        # noqa: E402
from orh import __version__, cosmic                           # noqa: E402

SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
INK, INK2, SURFACE, RULE = "#0b0b0b", "#52514e", "#fcfcfb", "#d9d8d2"
F_AXIS_FLOOR = 1e-5
REF_LABELS = {                       # what each reference class of src/orh/cosmic.py is, in words
    "moments": "uniform over observer-moments",
    "origins": "uniform over inhabited planets (each counted once, at its origin)",
    "late": "observer-moments weighted by the age of the planet's observer era",
}


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


def model_for(cfg, cutoff):
    g = cfg["grid"]
    return cosmic.PopulationModel(t_min_gyr=g["t_min_gyr"], t_max_gyr=g["t_max_gyr"], n_t=g["n_t"], n_m=g["n_m"],
                                  m_max=g["m_max"], t_first_gyr=g["t_first_gyr"], cutoff_gyr=cutoff)


def main(argv):
    figures_only = "--figures-only" in argv
    argv = [a for a in argv if a != "--figures-only"]
    cfg_path = (Path(argv[1]) if len(argv) > 1 else ROOT / "configs" / "orh_stage_e.json").resolve()
    out_dir = Path(argv[2]) if len(argv) > 2 else ROOT / "results"
    cfg = json.loads(cfg_path.read_text())
    if figures_only:
        redraw_figures(cfg, out_dir)
        return
    t0 = cfg["t_today_gyr"]
    figdir = ROOT / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    meta = dict(config=cfg, config_path=str(cfg_path.relative_to(ROOT)), orh_version=__version__,
                python=platform.python_version(), numpy=np.__version__, exact=False,
                note="a classical population model; every knob synthetic as named in src/orh/cosmic.py",
                **provenance(ROOT))
    models = {}
    curves = {}
    summary_rows = []
    with ResultWriter(cfg["name"] + "_summary", meta, out_dir=str(out_dir)) as out:
        for sc in cfg["scenarios"]:
            cutoff = sc["cutoff_gyr"]
            if cutoff not in models:
                models[cutoff] = model_for(cfg, cutoff)
            model = models[cutoff]
            L = np.inf if sc["L_obs"] is None else sc["L_obs"]
            for ref in cfg["reference_classes"]:
                for gamma in cfg["interaction_gammas"]:
                    for eps in cfg["exotic_eps"]:
                        w = model.weight(sc["m_min"], tau_evo=sc["tau_evo"], L_obs=L, ref_class=ref, gamma=gamma)
                        if eps > 0:
                            w = w + model.exotic_weight(eps)
                        s = model.summarize(w, t_today=t0)
                        row = dict(scenario=sc["name"], m_min=sc["m_min"], tau_evo_gyr=sc["tau_evo"],
                                   L_obs_gyr=sc["L_obs"], cutoff_gyr=cutoff, reference_class=ref, gamma=gamma,
                                   exotic_eps=eps, converged=s["converged"],
                                   F_today=s.get("F_today", float("nan")), t_peak_gyr=s.get("t_peak_gyr", float("nan")),
                                   t50_gyr=s.get("t50_gyr", float("nan")), t9999_gyr=s.get("t9999_gyr", float("nan")),
                                   T_support_gyr=s.get("T_support_gyr", float("nan")),
                                   tail_weight_last_tenth=s.get("tail_weight", float("nan")),
                                   grid_t_max_gyr=cfg["grid"]["t_max_gyr"], F_today_averaged=float("nan"))
                        out.write(row)
                        summary_rows.append(row)
                        if gamma == 0.0 and eps == 0.0:
                            curves[(sc["name"], ref)] = (model.t, s["p"], s["F"], w)
        # the two-history demonstration: pooled against averaged normalization
        demo = cfg["two_history_demo"]
        hs = []
        for name, p in zip(demo["histories"], demo["p"]):
            t, _, _, w = curves[(name, "moments")]
            hs.append(dict(t=t, w=w, p=p))
        both = cosmic.pooled_and_averaged_F(hs, t_today=t0)
        out.write(dict(scenario="two_history_demo:" + "+".join(demo["histories"]), m_min=None, tau_evo_gyr=None,
                       L_obs_gyr=None, cutoff_gyr=None, reference_class="moments", gamma=0.0, exotic_eps=0.0,
                       converged=True, F_today=both["pooled"], t_peak_gyr=float("nan"), t50_gyr=float("nan"),
                       t9999_gyr=float("nan"), T_support_gyr=float("nan"), tail_weight_last_tenth=float("nan"),
                       grid_t_max_gyr=cfg["grid"]["t_max_gyr"], F_today_averaged=both["averaged"]))

    # the curves on a coarse grid
    with ResultWriter(cfg["name"] + "_curves", meta, out_dir=str(out_dir)) as out:
        for (name, ref), (t, p, F, _) in curves.items():
            idx = np.unique(np.round(np.linspace(0, len(t) - 1, 300)).astype(int))
            for i in idx:
                out.write(dict(scenario=name, reference_class=ref, t_gyr=float(t[i]), dP_dlnt=float(t[i] * p[i]),
                               F=float(F[i])))

    # figure: dP/d ln t for the main scenarios, moments class
    main_names = ["loeb_baseline", "friendly_delay4.5", "friendly_delay4.5_L1", "suppressed_delay4.5",
                  "sunlike_delay4.5"]
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)
    for k, name in enumerate(main_names):
        t, p, F, _ = curves[(name, "moments")]
        dashed = name == "loeb_baseline"        # it lies under the friendly-delay curve on this scale
        ax.plot(t * 1e9, t * p, color=SERIES[k], linewidth=2.6 if dashed else 2, linestyle="--" if dashed else "-",
                label=name.replace("_", " ") + (" (under the next curve)" if dashed else ""))
    ax.axvline(t0 * 1e9, color=INK2, linewidth=1, linestyle=":")
    ax.text(t0 * 1e9 * 1.15, ax.get_ylim()[1] * 0.92, "today", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(1e8, 1e14)
    style(ax, "Where observer-moments fall in cosmic time (uniform over observer-moments)", "cosmic time (years)",
          "dP / d ln t")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(figdir / "orh_stage_e_w_of_t.png", dpi=160)
    plt.close(fig)

    figure_f_today(cfg, summary_rows, figdir, plt)
    print(f"wrote {out_dir / (cfg['name'] + '_summary.csv')}, {out_dir / (cfg['name'] + '_curves.csv')} and figures")


def figure_f_today(cfg, summary_rows, figdir, plt):
    """Figure 6: F(today) by scenario and reference class (gamma 0, eps 0). A value below the axis floor is drawn
    as an open marker sitting on the floor, with its own legend entry, so that no point drops off the axis (until
    2026-10-12 values were clamped at 1e-6 on an axis starting at 1e-5, which hid four of them)."""
    names = [sc["name"] for sc in cfg["scenarios"]]
    refs = cfg["reference_classes"]
    fig, ax = plt.subplots(figsize=(9.8, 5.2), facecolor=SURFACE)
    y = np.arange(len(names))
    any_below = False
    for k, ref in enumerate(refs):
        shown, below = [], []
        for i, name in enumerate(names):
            r = next(r for r in summary_rows if r["scenario"] == name and r["reference_class"] == ref
                     and float(r["gamma"]) == 0.0 and float(r["exotic_eps"]) == 0.0)
            v = float(r["F_today"])
            if not np.isfinite(v):
                continue                                    # no F where the integral does not converge
            (shown if v >= F_AXIS_FLOOR else below).append((v, y[i] + (k - 1) * 0.22))
        if shown:
            ax.scatter([v for v, _ in shown], [yy for _, yy in shown], color=SERIES[k], s=56,
                       label=REF_LABELS.get(ref, ref), zorder=3)
        if below:
            any_below = True
            ax.scatter([F_AXIS_FLOOR] * len(below), [yy for _, yy in below], s=70, marker="o", facecolors="none",
                       edgecolors=SERIES[k], linewidths=1.6, zorder=4, clip_on=False)
    if any_below:
        ax.scatter([], [], s=70, marker="o", facecolors="none", edgecolors=INK2, linewidths=1.6,
                   label=f"below the axis floor of 10^{int(np.log10(F_AXIS_FLOOR))} "
                         "(value in results/orh_stage_e_summary.csv)")
    ax.set_xscale("log")
    ax.set_xlim(F_AXIS_FLOOR, 1.2)
    ax.set_yticks(y)
    ax.set_yticklabels([n.replace("_", " ") for n in names], fontsize=8, color=INK2)
    ax.invert_yaxis()
    style(ax, "F(13.8 Gyr): the share of observer-moments at or before today, by scenario", "F(13.8 Gyr)", "")
    ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2)
    fig.tight_layout()
    fig.savefig(figdir / "orh_stage_e_f_today.png", dpi=160)
    plt.close(fig)


def redraw_figures(cfg, results_dir):
    """--figures-only: redraw figure 6 from the summary CSV already in results/, writing nothing under results/.
    The dP/d ln t figure is not redrawn: it is made from the fine time grid, which is not stored."""
    import csv
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    path = Path(results_dir) / (cfg["name"] + "_summary.csv")
    with path.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    figdir = ROOT / "docs" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    figure_f_today(cfg, rows, figdir, plt)
    print(f"redrew {figdir / 'orh_stage_e_f_today.png'} from {path}; results/ untouched")


if __name__ == "__main__":
    main(sys.argv)
