"""
Integration tests of notebooks/7_eigenvectors_of_G_per_scale.ipynb: the leading
eigenvectors of M_t(x), the integrand of G(x), at each noise level, for the frog and the
inception models of V1 and V4.

The reference is stored in tests/fixtures/per_scale/ (regenerate with
tests/fixtures/make_per_scale_fixtures.py):

    per_scale_data.npz
        data behind the figure of the frog: the image, tau = log t at the 24 noise
        levels, and for V1 and V4 the eigenvalues and the leading 4 eigenvectors
        (float16) at each level
    per_scale_reference.png   the notebook's figure of the frog rendered from that data

  - test_align_signs_*, test_compute_summary_* and test_*_plot_*
      run on CPU in CI: execute the notebook's own functions and plot cells (on the
      stored data) and compare the picture with the reference picture.
  - test_recompute_*
      run the notebook's setup, functions and compute cells on the real checkpoints,
      for the frog, and compare with the stored data. Skipped unless runs/ and the
      TinyImageNet data are present (not the case on GitHub).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

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
    MEAN_ABS_TOL,
    PER_SCALE_FIXTURES,
    REPO,
    image_difference,
    notebook7_functions,
    per_scale_inputs,
    render_per_scale_plot,
    run_notebook7_frog,
)

N_LEVELS, N_EIGVECS, P = 24, 4, 64 * 64

# The leading eigenvector is compared at the coarse noise levels only, where it is
# stable with 200 samples per level: two seeds agree with |cos| >= 0.94 at tau > 2 (V1
# and V4, test image 0), against 0.11 to 0.97 at finer levels.
COARSE_LOG_T, MIN_COS = 3.0, 0.9


def plot_difference(examples: dict, results: dict, tmp_path: Path) -> tuple:
    """Render the notebook's figure of the frog and compare it with the reference."""
    out = tmp_path / "per_scale.png"
    render_per_scale_plot(examples, results, out, tmp_path / "figures")
    reference = mpimg.imread(PER_SCALE_FIXTURES / "per_scale_reference.png")
    return image_difference(mpimg.imread(out), reference)


def test_fixture_is_wellformed():
    examples, results = per_scale_inputs()

    # the frog, 64 x 64
    assert tuple(examples["frog"][0].shape) == (1, 1, 64, 64)
    for area in AREAS:
        record = results[area]["frog"]

        # 24 noise levels, increasing, and 4 eigenpairs at each
        assert tuple(record.log_tau.shape) == (N_LEVELS,)
        assert bool((record.log_tau[1:] > record.log_tau[:-1]).all())
        assert tuple(record.eigenvalues.shape) == (N_LEVELS, N_EIGVECS)
        assert tuple(record.eigenvectors.shape) == (N_LEVELS, N_EIGVECS, P)

        # eigenvalues positive and decreasing at each level; unit eigenvectors
        assert bool((record.eigenvalues > 0).all())
        assert bool((record.eigenvalues[:, :-1] >= record.eigenvalues[:, 1:]).all())
        norms = record.eigenvectors.double().norm(dim=2).numpy()
        np.testing.assert_allclose(norms, 1.0, atol=1e-2)


def test_align_signs_flips_only_signs_to_agree_with_coarser_levels():
    align_signs = notebook7_functions()["align_signs"]

    # random eigenvectors of 6 levels, with random signs
    generator = torch.Generator().manual_seed(0)
    eigvecs = torch.randn(6, 3, 50, generator=generator)
    signs = torch.where(torch.rand(6, 3, 1, generator=generator) < 0.5, -1.0, 1.0)
    aligned = align_signs(eigvecs * signs)

    # only the signs change
    torch.testing.assert_close(aligned.abs(), eigvecs.abs())

    # each eigenvector agrees with the next coarser level; the coarsest sum to > 0
    assert bool(((aligned[:-1] * aligned[1:]).sum(dim=2) >= 0).all())
    assert bool((aligned[-1].sum(dim=1) >= 0).all())


def test_compute_summary_on_known_records():
    compute_summary = notebook7_functions()["compute_summary"]

    # three images and two noise levels: orthogonal leading eigenvectors at the first
    # level, the same one at the second; V4's leading eigenvalue larger at the first
    basis = torch.eye(4)
    results = {}
    for area, values in (("V1", [1.0, 1.0]), ("V4", [2.0, 0.5])):
        results[area] = {
            name: SimpleNamespace(
                log_tau=torch.tensor([0.0, 1.0]),
                eigenvalues=torch.tensor(values).view(2, 1),
                eigenvectors=torch.stack([basis[index], basis[3]]).view(2, 1, 4),
            )
            for index, name in enumerate(("a", "b", "c"))
        }
    summary = compute_summary(results, ["V1", "V4"])

    # |cos| 0 between two different images at the first level, 1 at the second
    torch.testing.assert_close(summary["similarity_V1"], torch.tensor([0.0, 1.0]))

    # V4 larger in every image at the first level, in none at the second
    torch.testing.assert_close(summary["second_larger"], torch.tensor([1.0, 0.0]))
    torch.testing.assert_close(summary["median_V4"], torch.tensor([2.0, 0.5]))


def test_notebook_plot_reproduces_reference(tmp_path):
    examples, results = per_scale_inputs()
    mean_abs, bad = plot_difference(examples, results, tmp_path)
    assert mean_abs < MEAN_ABS_TOL and bad < BAD_PIXEL_FRACTION, (
        f"the figure differs from the reference: mean |diff| = {mean_abs:.5f} (tol "
        f"{MEAN_ABS_TOL}), fraction of pixels off by > 0.1 = {bad:.5f} (tol "
        f"{BAD_PIXEL_FRACTION})"
    )

    # the plot cell also saved the figure in its folder
    assert (tmp_path / "figures" / "eigenvectors_per_scale_frog.png").exists()


def test_plot_comparison_detects_changes(tmp_path):
    """Guards against a vacuous comparison: changing the eigenvectors must fail it."""
    examples, results = per_scale_inputs()

    # reverse the order of the eigenvectors of V1 at every level
    record = results["V1"]["frog"]
    record.eigenvectors = record.eigenvectors.flip(1)
    mean_abs, bad = plot_difference(examples, results, tmp_path)
    assert mean_abs >= MEAN_ABS_TOL or bad >= BAD_PIXEL_FRACTION


def have_inputs() -> bool:
    """The checkpoints and the image the computation needs (not available on GitHub)."""
    needed = [REPO / "data/tiny-imagenet-200/train/n01644900/images/n01644900_0.JPEG"]
    for area in AREAS:
        needed += [
            REPO / f"runs/encoders/inception/{area}/best_model.pt",
            REPO / f"runs/diffusion/inception/{area}/final_model.pt",
        ]
    return all(path.exists() for path in needed)


@pytest.mark.slow
def test_recompute_matches_fixture():
    """
    Compare what the notebook computes for the frog with the stored data. M_t(x) is a
    seeded Monte-Carlo estimate: identical on the device the fixture was made on, close
    on another; only the leading eigenvector at coarse noise levels is stable enough to
    compare across devices, and the leading eigenvalue to within 30%.
    """
    if not have_inputs():
        pytest.skip("needs runs/ checkpoints and TinyImageNet (see README)")

    # the notebook's cells, for the frog, from the repository root
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        namespace = run_notebook7_frog()
    finally:
        os.chdir(cwd)

    # the same settings and noise levels as the fixture
    ref = np.load(PER_SCALE_FIXTURES / "per_scale_data.npz")
    assert int(ref["n_samples"]) == namespace["PARAMS"]["n_samples"], (
        "the notebook's n_samples changed: regenerate the fixtures"
    )
    for area in AREAS:
        record = namespace["results"][area]["frog"]
        np.testing.assert_allclose(record.log_tau.numpy(), ref["log_t"], atol=1e-5)

        # the leading eigenvector and eigenvalue at the coarse levels
        coarse = ref["log_t"] >= COARSE_LOG_T
        got = record.eigenvectors[coarse, 0].double()
        want = torch.from_numpy(ref[f"{area}_eigenvectors"][coarse, 0]).double()
        cos = (got * want).sum(dim=1).abs()
        assert bool((cos > MIN_COS).all()), f"{area}: |cos| per level = {cos.tolist()}"
        np.testing.assert_allclose(
            record.eigenvalues[coarse, 0].numpy(),
            ref[f"{area}_eigenvalues"][coarse, 0],
            rtol=0.3,
        )
