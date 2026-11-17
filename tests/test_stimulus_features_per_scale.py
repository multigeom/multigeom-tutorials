"""
Check that notebook 8 still computes and plots the stimulus features at a single noise
level, M_t^env(x), for the frog.

Integration tests of notebooks/8_stimulus_features_per_scale.ipynb: the leading
eigenvectors of the stimulus features M_t^env(x) at each noise level, from the
unconditional model of the inception V1 diffusion model, next to those of M_t(x) of V1
and V4, for the frog.

The reference is stored in tests/fixtures/stimulus_features/ (regenerate with
tests/fixtures/make_stimulus_features_fixtures.py):

    stimulus_features_data.npz
        the frog's stimulus features: tau = log t at the 24 noise levels, and the
        eigenvalues and the leading 4 eigenvectors (float16) at each level; M_t(x) of
        V1 and V4 is that of notebook 7's fixture (tests/fixtures/per_scale/)
    stimulus_features_reference.png
        the notebook's figure of the frog rendered from that data

  - test_*_plot_*
      run on CPU in CI: execute the notebook's own plot cell (with the functions of
      figures/per_scale.py, on the stored data) and compare the picture with the
      reference picture.
  - test_recompute_*
      run the notebook's setup and stimulus-features cells on the real checkpoints, for
      the frog, and compare with the stored data. Skipped unless runs/ and the
      TinyImageNet data are present (not the case on GitHub).
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
    BAD_PIXEL_FRACTION,
    FROG_IMAGE,
    MEAN_ABS_TOL,
    REPO,
    STIMULUS_FIXTURES,
    have_inputs,
    image_difference,
    render_stimulus_plot,
    run_notebook8_frog,
    stimulus_inputs,
)

# the notebooks these tests cover (orders the tables of tests/README.md)
pytestmark = pytest.mark.notebook(8)

N_LEVELS, N_EIGVECS, P = 24, 4, 64 * 64

# The leading eigenvector is compared at the coarse noise levels only, where it is
# stable: for the frog, two seeds agree with |cos| >= 0.93 and leading eigenvalues
# within 9% at tau >= 3, while the fine levels are not reproducible at all.
COARSE_LOG_T, MIN_COS = 3.0, 0.9


def plot_difference(examples: dict, env: dict, results: dict, tmp_path: Path) -> tuple:
    """Render the notebook's figure of the frog and compare it with the reference."""
    out = tmp_path / "stimulus_features.png"
    render_stimulus_plot(examples, env, results, out, tmp_path / "figures")
    reference = mpimg.imread(STIMULUS_FIXTURES / "stimulus_features_reference.png")
    return image_difference(mpimg.imread(out), reference)


def test_fixture_is_wellformed():
    """
    Verify that the stored stimulus features of notebook 8 (the frog) have the noise
    levels of notebook 7's fixture and 4 positive decreasing eigenvalues and unit-norm
    eigenvectors at each level.
    """
    examples, env, results = stimulus_inputs()
    record = env["frog"]

    # the noise levels of notebook 7's fixture, and 4 eigenpairs at each
    torch.testing.assert_close(record.log_tau, results["V1"]["frog"].log_tau)
    assert tuple(record.eigenvalues.shape) == (N_LEVELS, N_EIGVECS)
    assert tuple(record.eigenvectors.shape) == (N_LEVELS, N_EIGVECS, P)

    # eigenvalues positive and decreasing at each level; unit eigenvectors
    assert bool((record.eigenvalues > 0).all())
    assert bool((record.eigenvalues[:, :-1] >= record.eigenvalues[:, 1:]).all())
    norms = record.eigenvectors.double().norm(dim=2).numpy()
    np.testing.assert_allclose(norms, 1.0, atol=1e-2)


def test_notebook_plot_reproduces_reference(tmp_path):
    """
    Verify that the plot cell of notebook 8 draws the figure of the frog (stimulus
    features, then V1 and V4 from notebook 7's fixture) like the reference picture,
    and saves it.
    """
    mean_abs, bad = plot_difference(*stimulus_inputs(), tmp_path)
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"the figure differs from the reference: mean |diff| = {mean_abs:.5f} (tol "
        f"{MEAN_ABS_TOL}), fraction of pixels off by > 0.1 = {bad:.5f} (tol "
        f"{BAD_PIXEL_FRACTION})"
    )

    # the plot cell also saved the figure in its folder
    saved = tmp_path / "figures" / "stimulus_features_per_scale_frog.png"
    assert saved.exists()


def test_plot_comparison_detects_changes(tmp_path):
    """
    Verify that, with the four leading eigenvectors of the stimulus features
    at a single noise level (what stays uncertain about the image given its noisy
    version), M_t^env(x), drawn in reverse order (the 4th in place of the 1st) at
    every noise level, the figure of the frog differs from the reference picture
    beyond the tolerance of `test_notebook_plot_reproduces_reference` (mean
    |difference| ≥ 0.001, or ≥ 0.2% of the pixels off by more than 0.1), so that that
    test cannot pass by accident.
    """
    examples, env, results = stimulus_inputs()

    # reverse the order of the eigenvectors of the stimulus features at every level
    env["frog"].eigenvectors = env["frog"].eigenvectors.flip(1)
    mean_abs, bad = plot_difference(examples, env, results, tmp_path)
    assert mean_abs >= MEAN_ABS_TOL or bad >= BAD_PIXEL_FRACTION


@pytest.mark.slow
def test_recompute_matches_fixture():
    """
    Verify that the stimulus features of the frog recomputed from the V1 checkpoint
    have the stored settings and noise levels, and at the coarse levels (log noise
    variance τ ≥ 3) the stored leading eigenvector (absolute cosine |cos| > 0.9) and
    leading eigenvalue (within 30%).

    Compare the stimulus features that the notebook computes for the frog with the
    stored ones. They are a seeded Monte-Carlo estimate: identical on the device the
    fixture was made on, close on another; only the leading eigenvector at coarse
    noise levels is stable enough to compare across devices, and the leading
    eigenvalue to within 30%.
    """
    if not have_inputs([FROG_IMAGE], areas=("V1",)):
        pytest.skip("needs runs/ checkpoints and TinyImageNet (see README)")

    # the notebook's cells, for the frog, from the repository root
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        namespace = run_notebook8_frog()
    finally:
        os.chdir(cwd)

    # the same settings and noise levels as the fixture
    ref = np.load(STIMULUS_FIXTURES / "stimulus_features_data.npz")
    params = namespace["PARAMS"]
    stored = (int(ref["n_samples"]), int(ref["n_noisy"]), float(ref["eta"]))
    assert stored == (params["n_samples"], params["n_noisy"], params["eta"]), (
        "the notebook's settings changed: regenerate the fixtures"
    )
    record = namespace["env"]["frog"]
    np.testing.assert_allclose(record.log_tau.numpy(), ref["log_t"], atol=1e-5)

    # the leading eigenvector and eigenvalue at the coarse levels
    coarse = ref["log_t"] >= COARSE_LOG_T
    got = record.eigenvectors[coarse, 0].double()
    want = torch.from_numpy(ref["eigenvectors"][coarse, 0]).double()
    cos = (got * want).sum(dim=1).abs()
    assert bool((cos > MIN_COS).all()), f"|cos| per level = {cos.tolist()}"
    np.testing.assert_allclose(
        record.eigenvalues[coarse, 0].numpy(), ref["eigenvalues"][coarse, 0], rtol=0.3
    )
