"""Build the curated curled-torus reader package without changing raw results.

Requires the project's plots dependencies and Tectonic (default: executable on PATH).
The manifest hashes content; it excludes itself and disposable TeX auxiliary files.
"""
import argparse
import hashlib
import json
import platform
import re
import subprocess
import sys
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "docs/papers/curled_torus"
BASE = "5382053a60c4137cd446f83355e2a12177bb94ea"


def check_source_and_abstract():
    source = (PAPER / "paper.tex").read_text(encoding="utf-8")
    citations = {key for group in re.findall(r"\\cite\{([^}]+)\}", source)
                 for key in group.split(",")}
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", source))
    references = set(re.findall(r"\\ref\{([^}]+)\}", source))
    labels = set(re.findall(r"\\label\{([^}]+)\}", source))
    if citations-bibliography or references-labels:
        raise ValueError(("undefined citations/references", citations-bibliography, references-labels))
    abstract = source.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0].strip()
    # arXiv accepts TeX math. Keep the exact source wording; only reflow whitespace.
    (PAPER / "arxiv_abstract.txt").write_text(" ".join(abstract.split())+"\n", encoding="utf-8", newline="\n")


def manifest():
    files = list(PAPER.glob("*.tex")) + list(PAPER.glob("*.pdf"))
    files += [PAPER / "supplement.md", PAPER / "README.md", PAPER / "plain_language_README.md"] + list(PAPER.glob("*.py"))
    files += list(PAPER.glob("*_2026-10-09.log")) + [PAPER / "arxiv_abstract.txt"]
    files += [PAPER / name for name in (
        "endpoint_geometry_2026-10-11.json", "t60_wait_criterion_2026-10-11.json",
        "review_corrections_2026-10-11.md", "pytest_2026-10-11.log")]
    files += [ROOT / name for name in (
        "ASSUMPTIONS.md", "scripts/analyse_curled_revision.py", "scripts/build_curled_paper.py",
        "scripts/read_endpoint_geometry.py", "scripts/audit_t60_wait_criterion.py",
        "tests/test_endpoint_geometry.py", "tests/test_main_nerd_pages.py", "scripts/analyse_t60.py",
        "scripts/exact_torus_level.py", "scripts/analyse_paper_stats.py", "scripts/analyse_t38.py",
        "scripts/plot_paper_fig1.py", "scripts/plot_paper_fig_lambda.py", "scripts/plot_paper_fig_tori.py",
        "scripts/run_sealed_tube.py", "src/graphity/sealed.py", "tests/test_curled_revision.py",
        "docs/papers/measurement_scope.md", "src/graphity/dimension.py",
        "scripts/analyse_t9.py", "scripts/analyse_t10.py", "scripts/analyse_t22.py",
        "scripts/run_sealed_sheet_budget.py", "tests/test_t22.py", "tests/test_sealed_budget_thermometer.py",
        "docs/reading/notes/2026-10-09_prior_work_curled_torus.md",
        # revision of 10 October 2026: the retrospective reading of the decay rows, the second
        # prior-work search, and the interactive companion of the plain-language edition
        "scripts/read_wait_detector.py", "tests/test_read_wait_detector.py",
        "scripts/analyse_t58.py", "tests/test_t58.py", "scripts/make_t58_configs.py",
        "scripts/read_t58_sweeps.py", "tests/test_t58_sweeps.py",
        "scripts/run_tube_decay.py",
        # 10 October 2026, later: first exits counted by the move, and the long waits at lambda = 1.05 (T59)
        "scripts/run_first_exits.py", "scripts/analyse_t59.py", "scripts/make_t59_configs.py",
        "src/graphity/exits.py", "tests/test_first_exit_clocks.py", "tests/test_t59.py",
        "docs/reading/notes/2026-10-10_prior_work_lambda_above_one.md",
        "docs/public/curling_ladder_tube.html",
        # revision of 12 October 2026: retained heat (T61), the kinds of zero-energy end state, the count of
        # wrappings, and what the sealed runs' class "melted" can mean
        "scripts/run_burp_heat.py", "scripts/analyse_t61.py", "scripts/make_t61_configs.py",
        "tests/test_t61.py", "tests/test_t61_safeguards.py", "configs/t61_heldout_prediction.json",
        "scripts/read_flat_kind.py", "scripts/read_endpoint_width.py", "scripts/read_open_shells.py",
        "tests/test_read_shapes.py", "scripts/exact_flat_torus_widths.py", "tests/test_flat_torus_widths.py",
        "scripts/read_t9_squares.py", "tests/test_t9_squares.py")]
    # Normalize release text to the UTF-8/LF representation stored by Git.
    # PowerShell-redirection logs can otherwise be UTF-16 and Windows files CRLF.
    for path in set(files):
        if path.suffix != ".pdf":
            content = path.read_bytes()
            encoding = "utf-16" if content.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8"
            text = content.decode(encoding).replace("\r\n", "\n")
            path.write_text(text, encoding="utf-8", newline="\n")
    hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(set(files))}
    def installed_version(name):
        try:
            return version(name)
        except PackageNotFoundError:
            return None  # --manifest-only can retain existing, hashed figures without matplotlib

    payload = dict(base_source_and_data_commit=BASE, release_date="2026-10-12",
                   status="Reader manuscript and accompanying computational materials",
                   analysis_environment=dict(python=platform.python_version(),
                       packages={name: installed_version(name) for name in
                                 ("numpy", "scipy", "numba", "igraph", "networkx", "matplotlib")}),
                   provenance_limit="Original execution commits are not consistently recorded in run metadata",
                   sha256=hashes)
    (PAPER / "release_manifest.json").write_text(json.dumps(payload, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(f"Manifest: {len(hashes)} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tectonic", default="tectonic")
    parser.add_argument("--manifest-only", action="store_true")
    args = parser.parse_args()
    check_source_and_abstract()
    if not args.manifest_only:
        for name, figure in (("plot_paper_fig_tori", "fig_tori"),
                             ("plot_paper_fig1", "fig1"),
                             ("plot_paper_fig_lambda", "fig_lambda")):
            subprocess.run([sys.executable, str(ROOT / "scripts" / (name+".py")),
                            str(PAPER / (figure+".pdf"))], cwd=ROOT, check=True)
        compiler = subprocess.run([args.tectonic, "--version"], check=True,
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT).stdout
        run = subprocess.run([args.tectonic, "--keep-logs", "paper.tex"], cwd=PAPER,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = compiler.decode("utf-8", errors="replace")+run.stdout.decode("utf-8", errors="replace")
        (PAPER / "build_2026-10-09.log").write_text(
            output+f"\nTectonic exit code: {run.returncode}\n", encoding="utf-8", newline="\n")
        print(output)
        run.check_returncode()
        tex_log = (PAPER / "paper.log").read_text(encoding="utf-8", errors="replace")
        problems = [line for line in tex_log.splitlines()
                    if "Overfull" in line or ("undefined" in line.lower() and "warning" in line.lower())]
        if problems:
            raise RuntimeError("Manuscript layout/reference problems: " + repr(problems))
    manifest()


if __name__ == "__main__":
    main()
