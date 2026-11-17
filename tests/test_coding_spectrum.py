"""
Check that notebook 10 still computes and plots the coding spectrum, what the responses
take off the error of the denoiser at each noise level, and the information it adds up
to.

Integration tests of notebooks/10_coding_spectrum.ipynb: the coding spectrum ρ(τ) of the
inception V1 and V4 models, from responses to the 200 test images, under the envelope
spectrum, with the fine-end limit t · <trace J(x)>.

The reference is stored in tests/fixtures/coding_spectrum/ (regenerate with
tests/fixtures/make_coding_spectrum_fixtures.py):

    coding_spectrum_data.npz
        tau = log t at the 31 noise levels; for each area, the four terms of every test
        image (the errors without and with the responses, what the responses change,
        and notebook 7's estimate) and the trace of J(x) of every test image
    coding_spectrum_reference.png
        the notebook's figure rendered from that data

  - test_*_plot_*
      run on CPU in CI: execute the notebook's own plot cell (with the functions of
      figures/per_scale.py, on the stored data) and compare the picture with the
      reference picture.
  - test_recompute_*
      run the notebook's setup, spectrum and fine-end-limit cells on the real
      checkpoints and data, and compare with the stored data. Skipped unless runs/ and
      the TinyImageNet data are present (not the case on GitHub).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures.per_scale import compute_information  # noqa: E402
from nb_helpers import (  # noqa: E402
    AREAS,
    BAD_PIXEL_FRACTION,
    CODING_FIXTURES,
    CODING_TERMS,
    MEAN_ABS_TOL,
    REPO,
    coding_inputs,
    coding_record,
    have_inputs,
    image_difference,
    render_coding_plot,
    run_notebook10,
)

# the notebooks these tests cover (orders the tables of tests/README.md)
pytestmark = pytest.mark.notebook(10)

N_IMAGES, N_LEVELS = 200, 31

# Recomputed on another device (other noise draws and responses for the same images):
# the mean errors without and with the responses within 2% at each level; ½ ∫ ρ dτ
# within 1 nat, three times the largest standard deviation of the difference between
# two runs (√2 times the standard error over the images, at most 0.23 nats); the
# traces of J(x), which involve no random draw, within 0.1%. All are identical on the
# device the fixture was made on.
RECOMPUTE_RTOL = 0.02
INFORMATION_ATOL = 1.0
TRACE_RTOL = 1e-3


def plot_difference(spectra: dict, traces: dict, tmp_path: Path) -> tuple:
    """Render the notebook's figure and compare it with the reference."""
    out = tmp_path / "coding_spectrum.png"
    render_coding_plot(spectra, traces, out, tmp_path / "figures")
    reference = mpimg.imread(CODING_FIXTURES / "coding_spectrum_reference.png")
    return image_difference(mpimg.imread(out), reference)


def test_fixture_is_wellformed():
    """
    Verify that the stored coding spectrum of notebook 10 has 31 increasing noise
    levels and, for V1 and V4 and each of the 200 test images, the four terms of the
    curve at each level (positive errors without and with the responses, non-negative
    changes made by the responses) and a positive trace of the Fisher information of
    the responses J(x).
    """
    spectra, traces = coding_inputs()

    for area in AREAS:
        # 31 increasing noise levels, and the four terms of every image at each
        record = spectra[area]
        assert tuple(record.log_tau.shape) == (N_LEVELS,)
        assert bool((record.log_tau[1:] > record.log_tau[:-1]).all())
        for name in CODING_TERMS:
            assert tuple(getattr(record, name).shape) == (N_IMAGES, N_LEVELS)

        # squared errors are positive; what the responses change is never negative
        assert bool((record.env > 0).all()) and bool((record.coded > 0).all())
        assert bool((record.gain >= 0).all()) and bool((record.sampled >= 0).all())

        # a positive trace of J(x) per image
        assert tuple(traces[area].shape) == (N_IMAGES,)
        assert bool((traces[area] > 0).all())


def test_notebook_plot_reproduces_reference(tmp_path):
    """
    Verify that the plot cell of notebook 10 draws the coding spectrum like the
    reference picture, and saves it.
    """
    mean_abs, bad = plot_difference(*coding_inputs(), tmp_path)
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"the figure differs from the reference: mean |diff| = {mean_abs:.5f} (tol "
        f"{MEAN_ABS_TOL}), fraction of pixels off by > 0.1 = {bad:.5f} (tol "
        f"{BAD_PIXEL_FRACTION})"
    )

    # the plot cell also saved the figure in its folder
    assert (tmp_path / "figures" / "coding_spectrum.png").exists()


def test_plot_comparison_detects_changes(tmp_path):
    """
    Verify that, with V4's coding spectrum (what the responses take off the
    denoising error at each noise level), ρ(τ), doubled at every noise level (its
    error with the responses moved twice as far from the error without them), the
    coding spectrum figure differs from the reference picture beyond the tolerance of
    `test_notebook_plot_reproduces_reference` (mean |difference| ≥ 0.001, or ≥ 0.2% of
    the pixels off by more than 0.1), so that that test cannot pass by accident.
    """
    spectra, traces = coding_inputs()

    # V4's ρ made twice larger: its error with the responses moved twice as far from
    # the error without them
    record = spectra["V4"]
    coded = 2 * record.coded - record.env
    spectra["V4"] = coding_record(
        record.log_tau, record.env, coded, record.gain, record.sampled
    )
    mean_abs, bad = plot_difference(spectra, traces, tmp_path)
    assert mean_abs >= MEAN_ABS_TOL or bad >= BAD_PIXEL_FRACTION


@pytest.mark.slow
def test_recompute_matches_fixture():
    """
    Verify that the coding spectrum recomputed from the V1 and V4 checkpoints (about 7
    min on a GPU) has the stored noise levels, the stored mean errors without and with
    the responses (within 2%) and the mutual information between the responses and the
    image, ½∫ρ dτ (within 1 nat, both ways), and the encoders give the stored traces
    of the Fisher information of the responses J(x) (within 0.1%).

    Recompute the coding spectrum of both areas and the traces of J(x) from the
    checkpoints and the data (about 7 min on a GPU), and compare with the stored ones:
    the noise levels exactly, the mean errors and ½ ∫ ρ dτ (both ways) within their
    Monte-Carlo error, the traces to rounding.
    """
    if not have_inputs(["data/tiny-imagenet-200/val"]):
        pytest.skip("needs runs/ checkpoints and TinyImageNet (see README)")

    # the notebook's cells, from the repository root
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        namespace = run_notebook10()
    finally:
        os.chdir(cwd)

    # the noise levels, the mean errors, the information, and the traces of each area
    spectra, traces = coding_inputs()
    for area in AREAS:
        record, stored = namespace["spectra"][area], spectra[area]
        torch.testing.assert_close(record.log_tau, stored.log_tau)
        for name in ("env", "coded"):
            np.testing.assert_allclose(
                getattr(record, name).mean(dim=0).numpy(),
                getattr(stored, name).mean(dim=0).numpy(),
                rtol=RECOMPUTE_RTOL,
            )
        for curves in (lambda r: r.env - r.coded, lambda r: r.gain):
            information = compute_information(curves(record), record.log_tau)[0]
            expected = compute_information(curves(stored), stored.log_tau)[0]
            assert abs(information - expected) < INFORMATION_ATOL, area
        np.testing.assert_allclose(
            namespace["traces"][area].numpy(), traces[area].numpy(), rtol=TRACE_RTOL
        )
