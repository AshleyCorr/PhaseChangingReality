"""Build docs/public/curling_ladder.html (the owner's chart of 27 Sep) from the exact ladder and the run list. Usage:

    python scripts/build_curling_ladder.py

Reads docs/figures/ladder_kinds.json (scripts/exact_ladder_d.py: every kind of single move out of every rung) and
docs/figures/curling_ladder_runs.json (the experiments, their settings and outcomes, with the record that holds each),
adds each rung's energy per point as an exact straight line in lambda (computed on the built torus at lambda = 1 and 2),
and writes the page from docs/public/curling_ladder.template.html. Nothing is fitted.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from graphity.cqg_d import torus            # noqa: E402
from graphity.sealed_d import energy_d      # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def wall(r, lam):
    return min(-16 * k["dS"] + 4 * lam * k["dX"] for k in r["kinds"])


def main():
    kinds = json.loads((ROOT / "docs/figures/ladder_kinds.json").read_text(encoding="utf-8"))
    runs = json.loads((ROOT / "docs/figures/curling_ladder_runs.json").read_text(encoding="utf-8"))
    for r in kinds:
        adj, _ = torus(r["sides"])
        n = adj.shape[0]
        e1, e2 = energy_d(adj, 1.0) / n, energy_d(adj, 2.0) / n
        r["e1"], r["e0"] = e2 - e1, e1 - (e2 - e1)
        a_check = r["e1"] - 4 * r["curled"]
        assert abs(a_check) < 1e-9 and abs(r["e0"] + 4 * r["curled"]) < 1e-9, (r["D"], r["curled"], r["e0"], r["e1"])
    by = {(r["D"], r["curled"]): r for r in kinds}
    dims = sorted({r["D"] for r in kinds})
    flat = ", ".join("%d" % round(wall(by[(d, 0)], 1.25)) for d in dims if (d, 0) in by)
    cube = ", ".join("%g" % round(wall(by[(d, d)], 1.25), 1) for d in dims if (d, d) in by)
    pattern = ('<span class="ours">Exact</span> At λ = 1.25, flat space\'s wall for %s directions is %s: it doubles with each '
               'direction, so flat space gets stiffer as directions are added. The single cube\'s wall is %s. Every rung sits '
               '4(λ − 1) per point above the next, at every number of directions (checked on each built torus).'
               % (", ".join(map(str, dims)), flat, cube))
    d5 = [r for r in kinds if r["D"] == 5]
    d5_note = ("<span class=\"ours\">Caveat</span> The five-direction rungs use open sides of 6, and at 6 a short-side move "
               "can exist that longer tori lack (O49 found one at three directions). So their walls may be lower than on "
               "larger tori. Nothing has run at five directions; these are exact counts only." if d5 else
               "<span class=\"pending\">The five-direction rungs are still being computed.</span>")
    data = dict(kinds=kinds, runs=runs, pattern_note=pattern, d5_note=d5_note)
    tpl = (ROOT / "docs/public/curling_ladder.template.html").read_text(encoding="utf-8")
    assert tpl.count("/*__DATA__*/null") == 1
    out = tpl.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    (ROOT / "docs/public/curling_ladder.html").write_bytes(out.encode("utf-8"))
    print("rungs:", sorted(by))


if __name__ == "__main__":
    main()
