"""
Integration tests of notebooks/9_envelope_spectrum.ipynb: the envelope spectrum
ρ^env(τ), the error of the unconditional denoiser per unit of noise variance, on the
200 test images, for the null tokens of the inception V1 and V4 models, with the
Gaussian bound.

The reference is stored in tests/fixtures/envelope_spectrum/ (regenerate with
tests/fixtures/make_envelope_spectrum_fixtures.py):

    envelope_spectrum_data.npz
        tau = log t at the 31 noise levels, the curve of every test image for each
        model, the variances of the training images and the Gaussian bound
    envelope_spectrum_reference.png
        the notebook's figure rendered from that data

  - test_*_plot_*
      run on CPU in CI: execute the notebook's own plot cell (with the functions of
      figures/per_scale.py, on the stored data) and compare the picture with the
      reference picture.
  - test_recompute_*
      run the notebook's setup, spectrum and Gaussian-bound cells on the real
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

from nb_helpers import (  # noqa: E402
    AREAS,
    BAD_PIXEL_FRACTION,
    ENVELOPE_FIXTURES,
    MEAN_ABS_TOL,
    REPO,
    envelope_inputs,
    have_inputs,
    image_difference,
    render_envelope_plot,
    run_notebook9,
)

N_IMAGES, N_LEVELS, P = 200, 31, 64 * 64

# The mean curve over the 200 test images, recomputed on another device: within 2% (the
# Monte-Carlo error of 1600 noisy images per level is well below that).
RECOMPUTE_RTOL = 0.02


def plot_difference(spectra: dict, variances, gaussian, tmp_path: Path) -> tuple:
    """Render the notebook's figure and compare it with the reference."""
    out = tmp_path / "envelope_spectrum.png"
    render_envelope_plot(spectra, variances, gaussian, out, tmp_path / "figures")
    reference = mpimg.imread(ENVELOPE_FIXTURES / "envelope_spectrum_reference.png")
    return image_difference(mpimg.imread(out), reference)


def test_fixture_is_wellformed():
    spectra, variances, gaussian = envelope_inputs()

    # 31 increasing noise levels, a positive curve per image and model
    for area in AREAS:
        record = spectra[area]
        assert tuple(record.log_tau.shape) == (N_LEVELS,)
        assert bool((record.log_tau[1:] > record.log_tau[:-1]).all())
        assert tuple(record.values.shape) == (N_IMAGES, N_LEVELS)
        assert bool((record.values > 0).all())

    # the variances of the P pixels, largest first, and the Gaussian bound, which
    # decreases from about P to about tr C / t
    assert tuple(variances.shape) == (P,)
    assert bool((variances[:-1] >= variances[1:]).all()) and bool(
        (variances >= 0).all()
    )
    assert bool((gaussian[1:] < gaussian[:-1]).all()) and float(gaussian[0]) < P


def test_notebook_plot_reproduces_reference(tmp_path):
    mean_abs, bad = plot_difference(*envelope_inputs(), tmp_path)
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"the figure differs from the reference: mean |diff| = {mean_abs:.5f} (tol "
        f"{MEAN_ABS_TOL}), fraction of pixels off by > 0.1 = {bad:.5f} (tol "
        f"{BAD_PIXEL_FRACTION})"
    )

    # the plot cell also saved the figure in its folder
    assert (tmp_path / "figures" / "envelope_spectrum.png").exists()


def test_plot_comparison_detects_changes(tmp_path):
    """Guards against a vacuous comparison: a different V4 curve must fail it."""
    spectra, variances, gaussian = envelope_inputs()

    # the V4 model's error, made 1.5 times larger
    spectra["V4"].values = spectra["V4"].values * 1.5
    mean_abs, bad = plot_difference(spectra, variances, gaussian, tmp_path)
    assert mean_abs >= MEAN_ABS_TOL or bad >= BAD_PIXEL_FRACTION


@pytest.mark.slow
def test_recompute_matches_fixture():
    """
    Recompute the spectrum of both models and the Gaussian bound from the checkpoints
    and the data, and compare with the stored ones: the noise levels exactly, the mean
    curves within 2% (identical on the device the fixture was made on), the variances
    of the training images to rounding.
    """
    if not have_inputs(["data/tiny-imagenet-200/val", "data/tiny-imagenet-200/train"]):
        pytest.skip("needs runs/ checkpoints and TinyImageNet (see README)")

    # the notebook's cells, from the repository root
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        namespace = run_notebook9()
    finally:
        os.chdir(cwd)

    # the noise levels, the mean curve of each model, and the Gaussian bound
    spectra, variances, gaussian = envelope_inputs()
    for area in AREAS:
        record = namespace["spectra"][area]
        torch.testing.assert_close(record.log_tau, spectra[area].log_tau)
        np.testing.assert_allclose(
            record.values.mean(dim=0).numpy(),
            spectra[area].values.mean(dim=0).numpy(),
            rtol=RECOMPUTE_RTOL,
        )
    torch.testing.assert_close(namespace["variances"], variances, rtol=1e-6, atol=1e-9)
    torch.testing.assert_close(namespace["gaussian"], gaussian, rtol=1e-6, atol=0)
