"""
Unit tests of the helpers of notebooks 7 to 10 (figures/per_scale.py) on inputs whose
answer is known: no data, checkpoints or GPU, and no multigeom newer than the released
one.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from figures.per_scale import align_signs, compute_summary


def test_align_signs_flips_only_signs_to_agree_with_coarser_levels():
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


def test_compute_centre_on_known_curves():
    from figures.per_scale import compute_centre

    # a bump symmetric about log t = 3, and the same bump with a negative value at
    # log t = 0, which must not move the centre
    log_t = torch.arange(6, dtype=torch.float64)
    bump = torch.tensor([0.0, 0.0, 1.0, 2.0, 1.0, 0.0], dtype=torch.float64)
    dipped = torch.tensor([-5.0, 0.0, 1.0, 2.0, 1.0, 0.0], dtype=torch.float64)

    # both centred at 3
    assert abs(compute_centre(bump, log_t) - 3.0) < 1e-12
    assert abs(compute_centre(dipped, log_t) - 3.0) < 1e-12
