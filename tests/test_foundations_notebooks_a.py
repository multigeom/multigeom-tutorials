"""
Integration tests of the Foundations notebooks f1 to f4 (one neuron, a one-dimensional
stimulus, every quantity on a grid with `multigeom.lowdim`): each notebook runs end to
end on a CPU, and its key numbers match reference values.

Where each expected value comes from:
- closed forms of the series (Gaussian density peak, J = x^2 exp(-x^2) / sigma^2,
  Fisher distances 2 (f(0) - f(3)) / sigma, eigenvalues of the two-sigmoid code);
- converged-grid values of the series (mutual information 1.603, 2.419, 1.662 bits);
- the numbers stated in the source text of each notebook, given to 2 or 3 decimals,
  so the tolerance is half a unit of the last printed decimal unless stated;
- for the simulations, fixed random generators and tolerances of several standard
  errors.
No data or network is needed, so these tests run on GitHub.
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

NB_DIR = REPO / "notebooks"
SIGMA = 0.1

# The notebooks use a grid of 1,201 stimuli. Against a grid of 2,401 stimuli their
# mutual information moves by at most 0.0002 bits, and the converged values are given
# to 3 decimals (rounding 0.0005): 0.002 bits covers both with a margin.
MI_TOL = 2e-3
MI_CONVERGED = {"bell-shaped": 1.603, "monotonic twin": 2.419, "low-gain": 1.662}

# Fisher distance from -3 to 3: (1 / sigma) int |f'| = 2 (f(0) - f(3)) / sigma for the
# bell-shaped neuron and its twin, half of it for the low-gain neuron. The trapezoid
# rule on central differences with step 0.005 is within 0.002 of it.
D_BELL = 2 * (1 - math.exp(-4.5)) / SIGMA
D_TOL = 5e-3


def run(name: str) -> dict:
    """Run every code cell of a notebook in a fresh namespace and return it."""
    ns = run_all_code_cells({}, NB_DIR / name)
    plt.close("all")
    return ns


def test_f1_posterior_of_one_response():
    ns = run("f1_what_one_response_tells_you.ipynb")
    # closed forms: f(-1) = exp(-1/2), peak 1 / (sqrt(2 pi) sigma)
    assert ns["mean_shown"] == pytest.approx(math.exp(-0.5), abs=1e-12)
    assert ns["peak_density"] == pytest.approx(
        1 / (math.sqrt(2 * math.pi) * SIGMA), rel=1e-3
    )
    # 5,000 draws (fixed seed): standard error of a share near 0.68 is 0.0066
    assert ns["within_one_sigma"] == pytest.approx(0.683, abs=0.02)
    # source text: marginal highest near r = 0.08 at 1.49, 28.5% of stimuli below 0.1
    assert ns["marginal_peak_r"] == pytest.approx(0.08, abs=0.01)
    assert ns["marginal_peak"] == pytest.approx(1.49, abs=0.005)
    assert ns["share_below_tenth"] == pytest.approx(0.285, abs=0.002)
    # source text: two bumps at +-1.55 (0.26) and +-1.01 (0.20), one bump (0.50); the
    # half-widths are counted in grid steps of 0.005, hence 0.006 on the location
    expected = {0.3: (1.55, 0.26, 2), 0.6: (1.01, 0.20, 2), 1.0: (0.0, 0.50, 1)}
    for r_obs, (location, half_width, n_bumps) in expected.items():
        found = ns["posterior_bumps"][r_obs]
        assert len(found) == n_bumps
        for x_peak, mass, width in found:
            assert abs(x_peak) == pytest.approx(location, abs=0.006)
            assert mass == pytest.approx(1 / n_bumps, abs=1e-3)
            assert width == pytest.approx(half_width, abs=0.006)
    # closed form: likelihood ratio exp(-(1 - 0.6)^2 / (2 sigma^2)) = exp(-8)
    assert ns["likelihood_ratio"] == pytest.approx(math.exp(-8), rel=0.02)
    # sigmoid: one bump at ln(0.6 / 0.4)
    ((x_peak, mass, _),) = ns["sigmoid_bumps"]
    assert x_peak == pytest.approx(math.log(1.5), abs=0.003)
    assert mass == pytest.approx(1.0, abs=1e-6)


def test_f2_entropy_and_mutual_information():
    ns = run("f2_entropy_and_mutual_information.ipynb")
    assert ns["biased_coin"] == pytest.approx(0.469, abs=1e-3)
    # the grid covers a width of 6.005, so H(X) = log2 6.005, 0.001 above log2 6
    assert ns["prior_entropy"] == pytest.approx(math.log2(6.005), abs=1e-6)
    # source text, 2 decimals
    assert ns["h_post_bell"] == pytest.approx(0.54, abs=0.005)
    assert ns["h_post_twin"] == pytest.approx(-0.46, abs=0.005)
    assert ns["h_post_bell"] - ns["h_post_twin"] == pytest.approx(0.998, abs=5e-4)
    assert ns["twin_mass_right"] == pytest.approx(1.5e-4, abs=0.5e-5)
    assert ns["bump_sd"].item() == pytest.approx(0.18, abs=0.005)
    h_03, h_10 = ns["h_other"].tolist()
    assert h_03 == pytest.approx(1.15, abs=0.005)
    assert h_10 == pytest.approx(0.40, abs=0.005)
    for name, value in MI_CONVERGED.items():
        assert ns["mi_bits"][name] == pytest.approx(value, abs=MI_TOL)
    # the bell-shaped neuron leaves one bit about the side of the peak; the grid point
    # x = 0 counts as right of the peak, which moves the split by 1e-6 bits
    assert ns["side_left"]["bell-shaped"] == pytest.approx(1.0, abs=1e-4)
    assert ns["side_left"]["monotonic twin"] == pytest.approx(0.18, abs=0.005)
    # closed form of the noise entropy, and H(R) from the source text
    assert ns["noise_entropy"] == pytest.approx(
        0.5 * math.log2(2 * math.pi * math.e * SIGMA**2), abs=1e-12
    )
    for name, value in {
        "monotonic twin": 1.14,
        "low-gain": 0.39,
        "bell-shaped": 0.33,
    }.items():
        assert ns["response_entropy"][name] == pytest.approx(value, abs=0.005)
    # source text at sigma = 0.2
    assert ns["mi_doubled"]["bell-shaped"] == pytest.approx(0.94, abs=0.005)
    assert ns["mi_doubled"]["monotonic twin"] == pytest.approx(1.66, abs=0.005)


def test_f3_fisher_information():
    ns = run("f3_fisher_information.ipynb")
    assert ns["d_prime"]["flank"] == pytest.approx(2.35, abs=0.005)
    assert ns["d_prime"]["peak"] == pytest.approx(0.77, abs=0.005)
    # closed form: max of x^2 exp(-x^2) / sigma^2 is exp(-1) / sigma^2 at |x| = 1;
    # central differences with step 0.005 are within 1e-4 relative
    assert ns["J_max"] == pytest.approx(math.exp(-1) / SIGMA**2, rel=1e-3)
    assert ns["predicted"]["flank"] == pytest.approx(2.17, abs=0.005)
    assert ns["predicted"]["peak"] == pytest.approx(0.0, abs=1e-6)
    # 100,000 repeats: standard error of the mean score sqrt(J / n) = 0.019, of the
    # variance J sqrt(2 / n) = 0.16; tolerances of 3 standard errors
    assert ns["score_mean"] == pytest.approx(0.0, abs=0.06)
    assert ns["score_variance"] == pytest.approx(ns["J_probe"], abs=0.5)
    assert ns["J_probe"] == pytest.approx(math.exp(-1) / SIGMA**2, rel=1e-12)
    # closed form: KL of two Gaussians of equal width is d'^2 / 2
    assert ns["kl_flank"] == pytest.approx(ns["d_prime"]["flank"] ** 2 / 2, rel=1e-6)
    # Cramer-Rao at x = -1: nearly unbiased estimator, sd at or above the bound; the sd
    # of a sample sd from 100,000 repeats is about 0.2%, so 1% leaves room
    mean, sd, bound = ns["cramer_rao"][-1.0]
    assert mean == pytest.approx(-1.0, abs=0.005)
    assert bound == pytest.approx(SIGMA * math.exp(0.5), rel=1e-12)
    assert sd >= 0.99 * bound
    # source text: the band maps to -1.17 ... -0.83 and to -0.46 ... 0.46
    assert ns["flank_band"] == pytest.approx((-1.17, -0.83), abs=0.005)
    assert ns["peak_band"][0] == pytest.approx(-0.46, abs=0.005)
    # source text: population of eight neurons
    middle = ns["J_population"][ns["x"].abs() <= 1.2 + 1e-9]
    assert middle.min().item() == pytest.approx(108.9, abs=0.05)
    assert middle.max().item() == pytest.approx(110.8, abs=0.05)
    assert ns["J_at_2"] == pytest.approx(89.1, abs=0.05)
    # closed form: Poisson J = A x^2 exp(-x^2 / 2) is largest at sqrt(2) (step 0.005)
    assert ns["x_max_poisson"] == pytest.approx(math.sqrt(2), abs=0.005)


def test_f4_fisher_distance_and_two_dimensions():
    ns = run("f4_from_fisher_to_geometry.ipynb")
    # source text, 2 decimals
    for name, value in {"flank": 2.97, "peak": 0.62, "end": 0.33}.items():
        assert ns["step_lengths"][name] == pytest.approx(value, abs=0.005)
    # closed form: d_J(-1, 1) = 2 (1 - exp(-1/2)) / sigma, the same in any unit
    d_closed = 2 * (1 - math.exp(-0.5)) / SIGMA
    assert ns["d_minus1_to_1"] == pytest.approx(d_closed, abs=D_TOL)
    assert ns["d_in_new_units"] == pytest.approx(ns["d_minus1_to_1"], rel=1e-9)
    # closed form at the origin: 9 along each preferred direction, eigenvalues
    # 9 (1 +- cos 60) = 13.5 and 4.5, leading eigenvector at 30 degrees
    small, large = ns["eigenvalues"].tolist()
    assert (large, small) == pytest.approx((13.5, 4.5), rel=1e-9)
    assert ns["leading_angle"] == pytest.approx(30.0, abs=1e-6)
    assert ns["eigenvalues_parallel"].tolist() == pytest.approx([0.0, 18.0], abs=1e-9)
    expected = {"bell-shaped": D_BELL, "monotonic twin": D_BELL, "low-gain": D_BELL / 2}
    for name, value in expected.items():
        assert ns["fisher_distance"][name] == pytest.approx(value, abs=D_TOL)
        assert ns["mi_bits"][name] == pytest.approx(MI_CONVERGED[name], abs=MI_TOL)
