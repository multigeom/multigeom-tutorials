"""
Check the helpers shared by notebooks 7 to 10 (sign alignment, Gaussian bound,
integration) on inputs whose answer is known.

Unit tests of the helpers of notebooks 7 to 10 (figures/per_scale.py) on inputs whose
answer is known: no data, checkpoints or GPU, and no multigeom newer than the released
one.
"""

from __future__ import annotations

import pytest

import torch

from figures.per_scale import align_signs

# the notebooks these tests cover (orders the tables of tests/README.md)
pytestmark = pytest.mark.notebook(7, 8, 9, 10)


def test_align_signs_flips_only_signs_to_agree_with_coarser_levels():
    """
    Verify that `align_signs` (`figures/per_scale.py`, notebooks 7 and 8) only flips
    signs, makes each eigenvector agree with the same one at the next coarser level,
    and gives the coarsest a positive sum.
    """
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


def test_compute_gaussian_env_on_known_variances():
    """
    Verify that `compute_gaussian_env` (`figures/per_scale.py`, notebook 9), the
    envelope spectrum of a Gaussian prior, gives the sum over the directions of their
    variance c_i over c_i plus the noise variance t, Σ_i c_i / (c_i + t), on known
    variances: the number of directions that vary at tiny noise, ½ per unit-variance
    direction at t = 1, and the total variance over t, tr C / t, at huge noise.
    """
    from figures.per_scale import compute_gaussian_env

    # two directions of variance 1 and one of variance 0
    variances = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    log_t = torch.log(torch.tensor([1e-9, 1.0, 1e9], dtype=torch.float64))
    curve = compute_gaussian_env(variances, log_t)

    # tiny noise: the two directions with variance (a direction without any is known);
    # t = 1: 1/2 each; huge noise: the total variance over t
    expected = torch.tensor([2.0, 1.0, 2e-9], dtype=torch.float64)
    torch.testing.assert_close(curve, expected, rtol=1e-6, atol=0)


def test_compute_information_on_known_curves():
    """
    Verify that `compute_information` (`figures/per_scale.py`, notebook 10) gives half
    the area under the curve of each image (trapezoids over log t), the mean over the
    images and its standard error, on two piecewise-linear curves of known area.
    """
    from figures.per_scale import compute_information

    # two images whose curves are 1 and 3 between log t = 0 and 2, then fall linearly
    # to 0 at log t = 4: their areas are 3 and 9
    log_t = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    curves = torch.tensor([[1.0, 1.0, 0.0], [3.0, 3.0, 0.0]], dtype=torch.float64)
    mean, sem = compute_information(curves, log_t)

    # half of each area (1.5 and 4.5 nats), their mean, and its standard error (the
    # standard deviation of the two, 3 / √2, over √2)
    assert abs(mean - 3.0) < 1e-12
    assert abs(sem - 1.5) < 1e-12


def test_compute_normalized_on_known_values():
    """
    Verify that `compute_normalized` (`figures/per_scale.py`, notebooks 7, 8 and 10)
    divides the values by the largest value of the reference: a curve by its own
    maximum, and eigenvalues by the largest leading eigenvalue, so that the second ones
    keep their size relative to it.
    """
    import numpy as np

    from figures.per_scale import compute_normalized

    # a curve over its own maximum: its peak becomes 1, a negative value stays negative
    curve = np.array([-0.5, 1.0, 4.0, 2.0])
    np.testing.assert_allclose(compute_normalized(curve, curve), [-0.125, 0.25, 1, 0.5])

    # eigenvalues [levels, k] over the largest leading eigenvalue (first column)
    values = np.array([[2.0, 1.0], [8.0, 4.0], [4.0, 0.5]])
    expected = np.array([[0.25, 0.125], [1.0, 0.5], [0.5, 0.0625]])
    np.testing.assert_allclose(compute_normalized(values, values[:, 0]), expected)
