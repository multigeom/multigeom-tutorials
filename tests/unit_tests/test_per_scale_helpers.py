"""
Check the summary and integration helpers shared by notebooks 7 to 10 on inputs whose
answer is known.

Unit tests of the helpers of notebooks 7 to 10 (figures/per_scale.py) on inputs whose
answer is known: no data, checkpoints or GPU, and no multigeom newer than the released
one.
"""

from __future__ import annotations

import pytest
from types import SimpleNamespace

import torch

from figures.per_scale import align_signs, compute_summary

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


def test_compute_summary_on_known_records():
    """
    Verify that `compute_summary` (`figures/per_scale.py`, notebook 7) gives an
    absolute cosine |cos| of 0 between images with orthogonal eigenvectors and 1 with
    the same one, and the fraction of images where V4's leading eigenvalue is larger.
    """
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
