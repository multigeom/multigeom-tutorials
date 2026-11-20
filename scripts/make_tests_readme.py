"""
Write the tables of tests/README.md from the tests themselves: the first paragraph of
each test's docstring, its markers, and its folder (tests/ for the regression tests,
tests/unit_tests/ for the unit tests). Only the parts of the README between the marker
comments are rewritten; the rest is written by hand. Usage, from the repository root:

    python scripts/make_tests_readme.py           # rewrite the tables
    python scripts/make_tests_readme.py --check   # fail if they are out of date (CI)

It runs on every commit (pre-commit hook, see .pre-commit-config.yaml) and, with
--check, on GitHub Actions. It fails if a test has no docstring, or one that does not
start with "Verify that".
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import inspect
import io
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "tests" / "README.md"

# the markers of tests that GitHub Actions does not run (see .github/workflows)
NOT_ON_CI = ("slow", "gpu", "pretrained")

# the two tables: their marker comments in the README, and their folder in tests/
SECTIONS = {
    "regression-tests": lambda path: path.parent.name == "tests",
    "unit-tests": lambda path: path.parent.name == "unit_tests",
}

# unit tests of the multigeom package that check what the notebooks rely on: they are
# not in this repository, so they are listed here by hand (file, notebooks, test,
# summary, GitHub Actions) and added to the unit tests table
PACKAGE = "https://github.com/multigeom/multigeom/blob/main/tests/unit"
ENV_TESTS = f"{PACKAGE}/test_env.py"
PER_SCALE_TESTS = f"{PACKAGE}/test_per_scale_analysis.py"
PACKAGE_SUMMARIES = {
    ENV_TESTS: "Check that the estimator of the stimulus features at a single noise"
    " level, M_t^env(x), gives the known answer for a Gaussian prior and in two exact"
    " limits.",
    PER_SCALE_TESTS: "Check the analyses of the estimates at each noise scale (sign"
    " alignment, normalised curves, information, paired bootstrap), the Gaussian bound"
    " of the envelope spectrum and the Fisher information per neuron on inputs whose"
    " answer is known.",
}
NEXT_RELEASE = "in the package's checks (from its next release)"
PACKAGE_TESTS = [
    (
        ENV_TESTS,
        (8,),
        "test_eigenvalues_are_c_over_c_plus_t_not_squared",
        "Verify that, for a Gaussian prior with variances c_i whose answer is known,"
        " the four leading eigenvalues of the stimulus features at a single noise"
        " level, M_t^env(x), equal c_i / (c_i + t), with t the noise variance, at five"
        " noise levels (within 10%), and not their squares.",
        NEXT_RELEASE,
    ),
    (
        ENV_TESTS,
        (8,),
        "test_each_eigenvector_matches_its_own_prior_direction",
        "Verify that, for a Gaussian prior whose answer is known, each leading"
        " eigenvector of the stimulus features at a single noise level, M_t^env(x),"
        " points along its own direction of image variation, in the right order"
        " (absolute cosine |cos| above 0.95), at every noise level where neighbouring"
        " eigenvalues differ by at least 20%.",
        NEXT_RELEASE,
    ),
    (
        ENV_TESTS,
        (8,),
        "test_a_posterior_mean_that_ignores_x_t_gives_zero_rows",
        "Verify that, with a denoiser whose estimate of the clean image ignores the"
        " noisy image x_t, every row of probes that the estimator of the stimulus"
        " features draws is exactly zero, at each noise level: the stimulus features"
        " are then zero.",
        NEXT_RELEASE,
    ),
    (
        ENV_TESTS,
        (8,),
        "test_the_identity_gives_rows_equal_to_their_probes",
        "Verify that, with a denoiser that returns the noisy image unchanged, every"
        " row that the estimator of the stimulus features draws equals the random"
        " probe that produced it, at each noise level: the stimulus features are then"
        " the identity matrix.",
        NEXT_RELEASE,
    ),
    (
        ENV_TESTS,
        (8,),
        "test_the_identity_has_singular_values_near_one_with_many_probes",
        "Verify that, with the same denoiser that returns the noisy image unchanged and"
        " many probes, the eigenvalues that the estimator of the stimulus features"
        " returns are all close to 1, as for the identity matrix.",
        NEXT_RELEASE,
    ),
    (
        PER_SCALE_TESTS,
        (7, 8, 9, 10, 11),
        "test_align_signs_flips_only_signs_to_agree_with_coarser_levels",
        "Verify that `align_signs` (`multigeom.analysis`) only flips signs, makes each"
        " eigenvector agree with the same one at the next coarser level, and gives the"
        " coarsest a positive sum.",
        NEXT_RELEASE,
    ),
    (
        PER_SCALE_TESTS,
        (7, 8, 9, 10, 11),
        "test_compute_gaussian_env_on_known_variances",
        "Verify that `compute_gaussian_env` (`multigeom.geometry`), the envelope"
        " spectrum of a Gaussian prior, gives the sum over the directions of their"
        " variance c_i over c_i plus the noise variance t, Σ_i c_i / (c_i + t), on"
        " known variances: the number of directions that vary at tiny noise, ½ per"
        " unit-variance direction at t = 1, and the total variance over t, tr C / t, at"
        " huge noise.",
        NEXT_RELEASE,
    ),
    (
        PER_SCALE_TESTS,
        (7, 8, 9, 10, 11),
        "test_compute_information_on_known_curves",
        "Verify that `compute_information` (`multigeom.analysis`) gives half the area"
        " under the curve of each image (trapezoids over log t), the mean over the"
        " images and its standard error, on two piecewise-linear curves of known area.",
        NEXT_RELEASE,
    ),
    (
        PER_SCALE_TESTS,
        (7, 8, 9, 10, 11),
        "test_compute_normalized_on_known_values",
        "Verify that `compute_normalized` (`multigeom.analysis`) divides the values by"
        " the largest value of the reference: a curve by its own maximum, and"
        " eigenvalues by the largest leading eigenvalue, so that the second ones keep"
        " their size relative to it.",
        NEXT_RELEASE,
    ),
    (
        PER_SCALE_TESTS,
        (7, 8, 9, 10, 11),
        "test_compute_bootstrap_resamples_the_same_images_for_every_area",
        "Verify that `compute_bootstrap` (`multigeom.analysis`) draws the same images"
        " for every area in each resample (two areas with the same curves get the same"
        " resampled means), that curves equal for every image give a band of zero"
        " width, and that `compute_band` gives the 2.5th and 97.5th percentiles of the"
        " resamples.",
        NEXT_RELEASE,
    ),
    (
        PER_SCALE_TESTS,
        (7, 8, 9, 10, 11),
        "test_compute_fisher_per_cell_on_a_linear_encoder",
        "Verify that `compute_fisher_per_cell` (`multigeom.geometry`) gives, for an"
        " encoder whose responses are a known linear map A of the image, each neuron's"
        " pattern A_n / σ and Fisher information ‖A_n‖² / σ², and that their total J(x)"
        " has the eigenvalues of AᵀA / σ², whose sum is that of the neurons'.",
        NEXT_RELEASE,
    ),
]


class Collector:
    """A pytest plugin that keeps the collected tests."""

    def __init__(self):
        self.items = []

    def pytest_collection_finish(self, session):
        self.items = list(session.items)


def collect_tests() -> list:
    """The tests that pytest collects in tests/, in order, without running them."""
    # collect quietly (the list pytest prints is not needed)
    collector = Collector()
    with contextlib.redirect_stdout(io.StringIO()) as printed:
        code = pytest.main(
            ["--collect-only", "-q", "-p", "no:cacheprovider", str(ROOT / "tests")],
            plugins=[collector],
        )
    if code != 0:
        sys.exit(f"pytest could not collect the tests:\n{printed.getvalue()}")
    return collector.items


def get_summary(item) -> str | None:
    """The first paragraph of the test's docstring, on one line (None without one)."""
    doc = inspect.getdoc(item.function)
    if not doc:
        return None
    return " ".join(doc.split("\n\n")[0].split())


def get_ci_status(cases: list) -> str:
    """
    Whether GitHub Actions runs the test, from the markers of its cases (one, or one
    per parameter of a parametrized test).
    """
    # marked as not run on GitHub Actions
    names = [name for name in NOT_ON_CI if cases[0].get_closest_marker(name)]
    if names:
        return f"no ({', '.join(names)})"

    # run, but all or some of its cases skipped while what they need is missing
    skips = [case.get_closest_marker("skipif") for case in cases]
    reasons = list(
        dict.fromkeys(s.kwargs["reason"] for s in skips if s and s.kwargs.get("reason"))
    )
    if reasons and all(skips):
        return f"yes, but skipped while it {'; '.join(reasons)}"
    if reasons:
        return (
            "yes, but some of its cases are skipped while what they need is missing"
            f" ({'; '.join(reasons)})"
        )
    return "yes"


def get_notebooks(item) -> tuple:
    """The notebooks the test covers, from its `notebook` marker (none: empty)."""
    marker = item.get_closest_marker("notebook")
    return tuple(marker.args) if marker else ()


def get_rows(items: list) -> dict:
    """
    The rows of each table, one per test: (file, notebooks, test, summary, GitHub
    Actions), the files in the order of the notebooks they cover (the others last).
    """
    # the cases of each test function (several for a parametrized test), in order
    tests = {}
    for item in items:
        tests.setdefault((Path(item.path), item.originalname), []).append(item)

    # one row per test function
    rows, missing = {name: [] for name in SECTIONS}, []
    for (path, _), cases in tests.items():
        item = cases[0]

        # the summary from the docstring, and the table it belongs to
        summary = get_summary(item)
        if summary is None or not summary.startswith("Verify that"):
            missing.append(f"{path.relative_to(ROOT)}::{item.originalname}")
            continue
        section = next(name for name, inside in SECTIONS.items() if inside(path))
        file = path.relative_to(README.parent).as_posix()
        notebooks = get_notebooks(item)
        status = get_ci_status(cases)
        rows[section].append((file, notebooks, item.originalname, summary, status))
    if missing:
        sys.exit(
            'tests without a docstring that starts with "Verify that":\n  '
            + "\n  ".join(missing)
        )

    # the tests of the package, then all by notebook (a stable sort keeps the order of
    # the tests within a file)
    rows["unit-tests"] += PACKAGE_TESTS
    for section_rows in rows.values():
        section_rows.sort(key=lambda row: min(row[1], default=float("inf")))
    return rows


def get_file_summary(file: str) -> str:
    """
    The one-line summary of a test file: the first paragraph of its module docstring
    (for a file of the package, the entry of PACKAGE_SUMMARIES).
    """
    if file.startswith(PACKAGE):
        return PACKAGE_SUMMARIES[file]
    doc = ast.get_docstring(ast.parse((README.parent / file).read_text()))
    if not doc:
        sys.exit(f"tests/{file} has no module docstring to summarise it")
    return " ".join(doc.split("\n\n")[0].split())


def get_table(rows: list) -> str:
    """The count of the tests, and their table in Markdown."""
    # the counts of the tests of this repository: on GitHub Actions, not run there,
    # skipped while something is missing; then those of the package
    local = [row for row in rows if not row[0].startswith(PACKAGE)]
    n_off = sum(not status.startswith("yes") for *_, status in local)
    n_skipped = sum(status.startswith("yes, but") for *_, status in local)
    count = f"{len(local)} tests: {len(local) - n_off} run on GitHub Actions"
    if n_skipped:
        count += f" ({n_skipped} of them skipped while what they need is missing)"
    count += f", {n_off} do not." if n_off else "."
    if len(rows) > len(local):
        count += f" Plus {len(rows) - len(local)} of the multigeom package."

    # one table per file, under a heading with its notebooks and a link to it (its name
    # without folder, "test_" and ".py"); a skip reason becomes a note under the table
    lines = [count]
    for file, notebooks in dict.fromkeys((row[0], row[1]) for row in rows):
        name = Path(file).stem.removeprefix("test_")
        if file.startswith(PACKAGE):
            name += ", in the multigeom package"
        label = "Notebook" + ("s" if len(notebooks) > 1 else "")
        prefix = f"{label} {', '.join(map(str, notebooks))}: " if notebooks else ""
        lines += ["", f"#### {prefix}[{name}]({file})", "", get_file_summary(file), ""]
        lines += ["| Test | What it checks | GitHub Actions |", "|---|---|---|"]
        notes = []
        for _, _, test, summary, status in (row for row in rows if row[0] == file):
            if status.startswith("yes, but"):
                notes.append(status.removeprefix("yes, but "))
                status = "yes\\*"
            summary = summary.replace("|", "\\|")
            lines.append(f"| `{test.removeprefix('test_')}` | {summary} | {status} |")
        for note in dict.fromkeys(notes):
            lines += ["", f"\\* {note[0].upper()}{note[1:]}."]
    return "\n".join(lines)


def get_readme(text: str, rows: dict) -> str:
    """The README with the tables between their marker comments replaced."""
    for section, section_rows in rows.items():
        start, end = f"<!-- {section}:start -->", f"<!-- {section}:end -->"
        pattern = re.compile(re.escape(start) + r".*?" + re.escape(end), re.S)
        if not pattern.search(text):
            sys.exit(f"{README} has no {start} ... {end} block")
        block = f"{start}\n{get_table(section_rows)}\n{end}"
        text = pattern.sub(lambda match: block, text)
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the README is out of date"
    )
    args = parser.parse_args()

    # the README as the tests describe it
    current = README.read_text()
    expected = get_readme(current, get_rows(collect_tests()))

    # check it, or write it
    if expected == current:
        print(f"{README.relative_to(ROOT)} is up to date")
    elif args.check:
        sys.exit(
            f"{README.relative_to(ROOT)} is out of date: run "
            "python scripts/make_tests_readme.py and commit the result"
        )
    else:
        README.write_text(expected)
        print(f"updated {README.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
