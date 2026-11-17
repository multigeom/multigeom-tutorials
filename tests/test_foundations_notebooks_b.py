"""
Integration tests of the Foundations notebooks f5 to f8 (noise as a ruler, which stimuli
carry the information, the multi-scale Fisher information, the geometry gives back the
information): every code cell runs on a CPU, and the key numbers of each notebook match
their reference values. No data, checkpoint or network is needed.

Where the reference values come from:
- closed forms (Gaussian channel, Tweedie's formula, the split of the information of the
  two-bump prior), checked to rounding error;
- the values of the series computed on converged grids (1201 stimuli and 100 blur
  levels per decade for the three neurons, 4801 stimuli and 60 blur levels per decade
  for the locality test). The notebooks use coarser grids to run in seconds, and the
  tolerances below cover the measured difference (stated per test) and not more.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_helpers import REPO, run_all_code_cells  # noqa: E402

NB = REPO / "notebooks"
# the notebooks call plt.show(), which does nothing under the Agg backend
pytestmark = pytest.mark.filterwarnings("ignore:FigureCanvasAgg is non-interactive")
LOG2_17 = math.log2(17)


def _run(name: str) -> dict:
    """Run every code cell of a Foundations notebook; returns its namespace."""
    try:
        return run_all_code_cells({}, NB / name)
    finally:
        plt.close("all")


def test_f5_noise_as_a_ruler():
    ns = _run("f5_noise_as_a_ruler.ipynb")
    # closed forms of the two-bump prior: the peaks merge where sqrt(0.01 + t) = 1,
    # within the 0.001 step of the scan
    assert ns["merge_t"] == pytest.approx(0.99, abs=2e-3)
    # MMSE inside a bump at t = 0.003 is 0.01 t / (0.01 + t) up to the far bump
    assert ns["mmse_at"][0.003] == pytest.approx(0.01 * 0.003 / 0.013, rel=1e-3)
    # post values, quoted to 2 or 3 significant digits: tolerance half a unit of the
    # last quoted digit
    assert ns["mmse_at"][0.1] == pytest.approx(0.0124, abs=5e-5)
    assert ns["mmse_at"][3.0] == pytest.approx(0.748, abs=5e-4)
    assert ns["mmse_at"][100.0] == pytest.approx(1.00, abs=5e-3)
    assert list(ns["slopes"].values()) == pytest.approx([0.77, 0.09, 0.33], abs=5e-3)
    assert ns["denoised_one"] == pytest.approx(0.32, abs=5e-3)
    # Tweedie's formula is exact: the two denoisers agree to rounding error
    assert ns["tweedie_gap"] < 1e-12
    assert ns["score_at_point"].item() == pytest.approx(-4.54, abs=5e-3)
    for t, value in {0.001: 2.73, 0.01: 1.50, 0.1: 1.06, 0.3: 0.89, 10.0: 0.07}.items():
        assert ns["info_at"][t] == pytest.approx(value, abs=5e-3)
    # the integral of the denoising error up to t = 1e4 misses about 1e-4 bits
    assert ns["info_from_mmse"] == pytest.approx(ns["info_at"][0.1], abs=3e-4)
    assert ns["contribution_min"][1] == pytest.approx(0.088, abs=5e-4)
    # positions of the extremes: one step of the 401-level grid (a factor 1.03)
    assert ns["contribution_min"][0] == pytest.approx(0.090, rel=0.03)
    assert ns["contribution_max"][0] == pytest.approx(0.64, rel=0.03)
    assert ns["contribution_max"][1] == pytest.approx(0.35, abs=5e-3)
    for t, value in {0.1: 0.99, 0.3: 0.86, 1.0: 0.48, 3.0: 0.21, 10.0: 0.07}.items():
        assert ns["label_info_at"][t] == pytest.approx(value, abs=5e-3)
    # the split into which bump and where in the bump is exact
    assert ns["split_gap"] < 1e-10
    # the which-bump area is 1 bit; 1e-3 is lost outside t = 0.001 to 1000
    assert ns["label_area"] == pytest.approx(1.0, abs=2e-3)


def test_f6_which_stimuli_carry_the_information():
    ns = _run("f6_which_stimuli_carry_the_information.ipynb")
    at = ns["at"]
    # specific surprise and SSI do not depend on the blur levels: the 301-point grid
    # gives the post values within half a unit of the last quoted digit
    assert ns["mi_bell"] == pytest.approx(1.603, abs=1e-3)
    assert at(ns["surprise_bell"], 0) == pytest.approx(2.34, abs=5e-3)
    assert at(ns["surprise_bell"], -1) == pytest.approx(2.04, abs=5e-3)
    assert at(ns["ssi_bell"], 0) == pytest.approx(2.21, abs=5e-3)
    assert at(ns["ssi_bell"], -1) == pytest.approx(1.98, abs=5e-3)
    largest = ns["largest_blurred_fisher"]
    assert largest[0.0] == pytest.approx(36.8, abs=0.05)
    assert largest[0.3] == pytest.approx(2.8, abs=0.05)
    assert largest[3.0] == pytest.approx(0.084, abs=5e-4)
    # identity: measured gap 0.004 bits on this grid, < 0.001 on the converged grid
    assert ns["info_from_fisher"] == pytest.approx(ns["mi_bell"], abs=0.01)
    # local information: converged values; this grid is 0.002 to 0.012 bits higher
    local, twin = ns["local_bell"], ns["local_twin"]
    for values, x0, ref in (
        (local, -1, 2.370),
        (local, 0, 1.044),
        (local, -3, 0.574),
        (twin, 0, 2.51),
        (twin, -1, 3.31),
    ):
        assert at(values, x0) == pytest.approx(ref, abs=0.015)
    assert local.max().item() == pytest.approx(2.392, abs=0.015)
    # locality test (601 stimuli, 10 levels per decade); converged values in bits
    measures, y = ns["measures"], ns["y"]
    reference = {
        "one": (0.550, 1.486, 4.994, 4.667),
        "c = 6": (1.150, 1.314, 3.467, 3.734),
        "c = 10": (1.096, 1.361, 3.582, 3.711),
        "c = 15": (0.911, 1.414, 3.994, 3.667),
    }
    for name, (mi, local_peak, surprise_peak, ssi_peak) in reference.items():
        m = measures[name]
        assert m["MI"] == pytest.approx(mi, abs=1e-3)
        # measured: local information at x = -3 is 0.007 to 0.008 bits high
        assert at(m["local"], -3, y) == pytest.approx(local_peak, abs=0.012)
        assert at(m["surprise"], -3, y) == pytest.approx(surprise_peak, abs=2e-3)
        assert at(m["SSI"], -3, y) == pytest.approx(ssi_peak, abs=2e-3)
    # losses at x = -3: measured within 0.002 bits of the converged ones
    for name, (local_loss, surprise_loss, ssi_loss) in {
        "c = 6": (0.172, 1.526, 0.932),
        "c = 10": (0.125, 1.411, 0.956),
        "c = 15": (0.072, 1.000, 1.000),
    }.items():
        loss = ns["loss_at_peak"][name]
        assert loss["local"] == pytest.approx(local_loss, abs=4e-3)
        assert loss["surprise"] == pytest.approx(surprise_loss, abs=3e-3)
        assert loss["SSI"] == pytest.approx(ssi_loss, abs=3e-3)
    shares = ns["loss_shares"]
    assert shares["below t = 3"] < 0.01
    assert shares["above t = 10"] == pytest.approx(0.97, abs=0.01)
    assert shares["above t = 100"] == pytest.approx(0.30, abs=0.02)
    # Try it: the loss of the local information falls with the distance of the bump
    assert ns["sweep"]["local"] == sorted(ns["sweep"]["local"], reverse=True)


def test_f7_multiscale_fisher_information():
    ns = _run("f7_multiscale_fisher_information.ipynb")
    at, g, d = ns["at"], ns["g"], ns["distances"]
    # mutual information: converged values; this grid is within 0.004 bits
    for name, ref in {
        "bell-shaped": 1.603,
        "monotonic twin": 2.419,
        "low-gain": 1.662,
    }.items():
        assert ns["mi_bits"][name] == pytest.approx(ref, abs=5e-3)
    # G: converged values; this grid is 0.004 to 0.012 higher
    for name, (g_flank, g_peak) in {
        "bell-shaped": (3.286, 1.448),
        "monotonic twin": (4.589, 3.482),
        "low-gain": (3.049, 2.564),
    }.items():
        assert at(g[name], -1) == pytest.approx(g_flank, abs=0.015)
        assert at(g[name], 0) == pytest.approx(g_peak, abs=0.015)
    # distances: the brief's reference values; G distances are 0.01 to 0.02 high here
    for name, (dj, dg, dj11, dg11) in {
        "bell-shaped": (19.78, 8.78, 7.87, 3.06),
        "monotonic twin": (19.78, 10.83, 7.87, 4.04),
        "low-gain": (9.89, 9.00, 3.93, 3.36),
    }.items():
        assert d[name]["d_J(-3, 3)"] == pytest.approx(dj, abs=5e-3)
        assert d[name]["d_G(-3, 3)"] == pytest.approx(dg, abs=0.025)
        assert d[name]["d_J(-1, 1)"] == pytest.approx(dj11, abs=5e-3)
        assert d[name]["d_G(-1, 1)"] == pytest.approx(dg11, abs=0.015)
    # the order of the G distances is the order of the bits
    order = sorted(d, key=lambda n: d[n]["d_G(-3, 3)"])
    assert order == sorted(ns["mi_bits"], key=ns["mi_bits"].get)
    # shares of G above t = 1 (post: 12%, 30%, 25%, 41%), measured within 0.01
    shares = ns["share_above_1"]
    assert shares[("bell-shaped", -1)] == pytest.approx(0.12, abs=0.01)
    assert shares[("monotonic twin", -1)] == pytest.approx(0.30, abs=0.01)
    assert shares[("bell-shaped", 0)] == pytest.approx(0.25, abs=0.01)
    assert shares[("monotonic twin", 0)] == pytest.approx(0.41, abs=0.012)
    # change of unit: G unchanged, G distance times c, Fisher distance unchanged
    assert ns["unit_gap"] < 1e-3
    c = ns["UNIT"]
    assert ns["distance"](ns["y"], ns["g_y"], -3 * c, 3 * c).item() == pytest.approx(
        c * d["bell-shaped"]["d_G(-3, 3)"], rel=1e-3
    )


def test_f8_geometry_gives_back_the_information():
    ns = _run("f8_geometry_gives_back_the_information.ipynb")
    # three neurons: measured gap 0.004 to 0.005 bits on this grid
    for name, (from_geometry, direct) in ns["identity"].items():
        assert from_geometry == pytest.approx(direct, abs=0.01), name
    # closed forms of the Gaussian channel
    along, across = ns["gaussian"]["along"], ns["gaussian"]["across"]
    assert along["I bits"] == pytest.approx(
        0.5 * LOG2_17 + 0.5 * math.log2(1.0625), abs=1e-12
    )
    assert along["half trace bits"] == pytest.approx(along["I bits"], abs=1e-12)
    assert along["G eigenvalues"] == pytest.approx(
        [math.log(17), math.log(1.0625)], abs=1e-12
    )
    assert across["I bits"] == pytest.approx(1.0, abs=1e-12)
    assert across["G eigenvalues"] == pytest.approx([math.log(2)] * 2, abs=1e-12)
    # nearest-neighbour form, fixed seed: within 4 standard errors of the mean of the
    # draws plus 0.01 bits, the largest bias measured for N between 300 and 1000 in
    # runs with many more draws; and the spread falls with N
    for n, draws in ns["estimates"].items():
        standard_error = draws.std().item() / math.sqrt(len(draws))
        assert abs(draws.mean().item() - along["I bits"]) < 4 * standard_error + 0.01, n
        assert ns["long_axis_share"][n] == pytest.approx(0.5, abs=0.03)
    spreads = [ns["estimates"][n].std().item() for n in ns["SAMPLE_SIZES"]]
    assert spreads[-1] < spreads[0] / 4
    # Try it: the code turned by 45 degrees (post: 1.73 bits, G at 15, J at 45 degrees)
    assert ns["info_45"] == pytest.approx(1.73, abs=5e-3)
    assert ns["leading_angle"](ns["g_45"]) == pytest.approx(15, abs=0.5)
    assert ns["leading_angle"](ns["fisher_45"]) == pytest.approx(45, abs=1e-6)
