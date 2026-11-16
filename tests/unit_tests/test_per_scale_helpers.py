"""
Unit tests of the helpers of notebooks 7 and 8 (figures/per_scale.py) on inputs whose
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
