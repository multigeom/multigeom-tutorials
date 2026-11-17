"""
Check that every eigenvector the notebooks plot is a true eigenvector of its matrix,
with the right eigenvalue.

Unit tests that every eigenvector the notebooks plot is an eigenvector of the matrix it
belongs to, for each estimator: the Fisher information of the responses J(x) (notebooks
0, 1, 3), the multiscale Fisher information G(x) (notebooks 2 to 4), G(x) at each noise
level, M_t(x) (notebook 7), and the stimulus features at each noise level, M_t^env(x)
(notebook 8).

Each of these matrices is M = DᵀD for a matrix of rows D (differences of noise
predictions, finite-difference probes, or the Jacobian of the encoder), and the
estimators return the top eigenpairs (λ_i, v_i) from the SVD of D. For each pair, the
relative residual ‖M v_i − λ_i v_i‖ / ‖M‖ must be about 0 and ‖v_i‖ about 1, with
M v_i = Dᵀ(D v_i) computed from the same rows D that the estimator used, without
building M. This catches a wrong index into the SVD (left instead of right singular
vectors, eigenvalues paired with the wrong vectors) and a wrong reshape.
"""

from __future__ import annotations

import importlib

import pytest
import torch
import torch.nn as nn

import multigeom
from estimator_mocks import (
    PARAMS,
    RecordingDenoiser,
    ResponseBlindDenoiser,
    compute_deltas,
    get_alphas_cumprod,
    get_cfg,
    get_levels,
    get_noise_variance,
    run_estimator,
)

# the notebooks these tests cover (orders the tables of tests/README.md)
pytestmark = pytest.mark.notebook(0, 1, 2, 3, 4, 7, 8)

# the estimators of G(x) and of the stimulus features at each noise level are newer than
# the multigeom on PyPI: their tests are skipped until it has them
compute_g_per_scale = getattr(multigeom, "compute_g_per_scale", None)
compute_env_per_scale = getattr(multigeom, "compute_env_per_scale", None)

# the eigenpairs checked per matrix, the response noise σ of the Fisher information, and
# the largest relative residual allowed (the SVD is in float32: residuals ~1e-7)
TOP_K = 10
SIGMA = 0.22
RESIDUAL_TOL = 1e-5


def get_residuals(
    rows: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    squared: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    For each eigenpair (λ_i, v_i) of M = DᵀD, with D the `rows` [N, P] and the
    `eigenvectors` [k, P]: the relative residual ‖M v_i − λ_i v_i‖ / ‖M‖ (‖M‖ its
    largest eigenvalue), with M v_i = Dᵀ(D v_i), and the norm ‖v_i‖. When the
    eigenvalues are the singular values of D (`squared` False), those of M are λ_i².
    """
    # in float64: the rows, the eigenvectors, and the eigenvalues of M
    rows, vectors = rows.double().flatten(1), eigenvectors.double()
    values = eigenvalues.double() if squared else eigenvalues.double() ** 2

    # M v_i = Dᵀ(D v_i) for every i at once, [k, P], and the norm of M, σ_max(D)²
    applied = (rows @ vectors.T).T @ rows
    norm = torch.linalg.matrix_norm(rows, ord=2) ** 2

    # the residuals, relative to ‖M‖, and the norms of the eigenvectors
    residuals = (applied - values[:, None] * vectors).norm(dim=1) / norm
    return residuals, vectors.norm(dim=1)


def assert_eigenpairs(
    rows: torch.Tensor,
    eigenvalues: torch.Tensor,
    eigenvectors: torch.Tensor,
    squared: bool = True,
) -> None:
    """Every eigenpair is one of DᵀD (residual ≈ 0), with a unit eigenvector."""
    residuals, norms = get_residuals(rows, eigenvalues, eigenvectors, squared)
    assert float(residuals.max()) < RESIDUAL_TOL, residuals
    torch.testing.assert_close(norms, torch.ones_like(norms), rtol=0, atol=1e-5)


def get_level_rows(deltas: dict, alphas_cumprod: torch.Tensor) -> dict:
    """
    The rows D of G(x) at each noise level, M_t = DᵀD, from the differences of the two
    denoised images of each sample δ_s: δ_s / √(2 N t), N the number of samples and t
    the noise variance of the level ({DDPM step: [N, P]}).
    """
    rows = {}
    for level, level_deltas in deltas.items():
        noise_variance = get_noise_variance(alphas_cumprod, level)
        rows[level] = level_deltas / (2 * len(level_deltas) * noise_variance) ** 0.5
    return rows


def get_trapezoid_weights(alphas_cumprod: torch.Tensor) -> torch.Tensor:
    """
    The trapezoid weights w_j of G(x) = Σ_j w_j M_tj over the log noise variance
    τ_j = log t_j of the noise levels (half weights at both ends).
    """
    # the log noise variance of each level, as the estimator computes it (float32)
    alpha_bar = alphas_cumprod[get_levels()]
    log_tau = torch.log((1 - alpha_bar) / alpha_bar).double()

    # half the distance between the two neighbours; half the step at both ends
    weights = torch.zeros_like(log_tau)
    weights[0] = 0.5 * (log_tau[1] - log_tau[0])
    weights[-1] = 0.5 * (log_tau[-1] - log_tau[-2])
    weights[1:-1] = 0.5 * (log_tau[2:] - log_tau[:-2])
    return weights


class LinearEncoder(nn.Module):
    """
    A mock encoder whose responses are a fixed linear map A of the image, built like
    the reference encoders (input conversion, backbone, readout), so that the Jacobian
    of the responses is A.
    """

    def __init__(self):
        super().__init__()

        # the map A from the pixels to the neurons, fixed
        n_pixels = PARAMS["image_size"] ** 2
        generator = torch.Generator().manual_seed(3)
        weight = torch.randn(PARAMS["n_neurons"], n_pixels, generator=generator)
        self.register_buffer("weight", weight)

        self.n_neurons = PARAMS["n_neurons"]
        self.backbone = nn.Flatten()

    def _to_backbone_input(self, images: torch.Tensor) -> torch.Tensor:
        return images

    def readout(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.weight.T

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.readout(self.backbone(self._to_backbone_input(images)))


def get_j_case(monkeypatch) -> list[tuple]:
    """
    The Fisher information of the responses J(x) = DᵀD of a random image, from
    `compute_fisher` with a mock linear encoder: D is its Jacobian A over the response
    noise σ. Returns [(rows, eigenvalues, eigenvectors, squared)].
    """
    # J(x) of a random image, all its eigenpairs (one per neuron)
    encoder = LinearEncoder()
    size = PARAMS["image_size"]
    image = torch.rand(1, 1, size, size, generator=torch.Generator().manual_seed(0))
    out = multigeom.compute_fisher(
        image=image,
        encoder=encoder,
        sigma=SIGMA,
        top_k=encoder.n_neurons,
        device=torch.device("cpu"),
    )
    return [(encoder.weight / SIGMA, out.eigenvalues, out.eigenvectors, True)]


def get_g_case(monkeypatch) -> list[tuple]:
    """
    The multiscale Fisher information G(x) = DᵀD of a random image, from `compute_g`
    with a mock denoiser that records its calls: D stacks the rows of every noise level
    (see `get_level_rows`), each weighted by √w_j, w_j the trapezoid weight of the
    level. Returns [(rows, eigenvalues, eigenvectors, squared)].
    """
    # G(x), and the rows of all the levels rebuilt from the calls, weighted and stacked
    denoiser = RecordingDenoiser(get_alphas_cumprod())
    out = run_estimator(multigeom.compute_g, denoiser, top_k=TOP_K)
    alphas_cumprod = denoiser.alphas_cumprod
    rows = get_level_rows(
        compute_deltas(denoiser.calls, alphas_cumprod), alphas_cumprod
    )
    weights = get_trapezoid_weights(alphas_cumprod)
    stacked = torch.cat(
        [weight.sqrt() * rows[level] for weight, level in zip(weights, get_levels())]
    )
    return [(stacked, out.eigenvalues, out.eigenvectors, True)]


def get_m_t_case(monkeypatch) -> list[tuple]:
    """
    The multiscale Fisher information at each noise level, M_t(x) = DᵀD, of a random
    image, from `compute_g_per_scale` with a mock denoiser that records its calls: D
    holds the rows of that level (see `get_level_rows`). Returns one (rows, eigenvalues,
    eigenvectors, squared) per noise level.
    """
    # M_t(x) at each level, and the rows of each level rebuilt from the calls
    denoiser = RecordingDenoiser(get_alphas_cumprod())
    out = run_estimator(compute_g_per_scale, denoiser, top_k=TOP_K)
    alphas_cumprod = denoiser.alphas_cumprod
    rows = get_level_rows(
        compute_deltas(denoiser.calls, alphas_cumprod), alphas_cumprod
    )
    return [
        (rows[level], out.eigenvalues[index], out.eigenvectors[index], True)
        for index, level in enumerate(get_levels())
    ]


def get_env_case(monkeypatch) -> list[tuple]:
    """
    The stimulus features at each noise level, M_t^env(x), of a random image, from
    `compute_env_per_scale` with the null token of a mock denoiser: D holds the
    finite-difference probes of that level, taken from the estimator on their way to the
    SVD, and DᵀD ≈ M_t^env², so the eigenvalues returned are the singular values of D
    (squared False). Returns one (rows, eigenvalues, eigenvectors, squared) per level.
    """
    # keep the rows of each level on their way to the SVD
    env = importlib.import_module("multigeom.geometry.env")
    original, captured = env._per_scale_output, []

    def keep_rows(log_tau, levels, k, sampler, squared=True):
        levels = [rows.detach().clone() for rows in levels]
        captured.extend(levels)
        return original(log_tau, iter(levels), k, sampler, squared)

    monkeypatch.setattr(env, "_per_scale_output", keep_rows)

    # the stimulus features of a random image, from the null token of a mock denoiser
    size = PARAMS["image_size"]
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(1, 1, size, size, generator=generator) * 2 - 1
    cfg = get_cfg()
    cfg.top_k = TOP_K
    denoiser = ResponseBlindDenoiser(get_alphas_cumprod())
    denoiser.n_neurons = PARAMS["n_neurons"]  # the size of its null token
    out = compute_env_per_scale(
        image=image,
        unet=denoiser,
        alphas_cumprod=get_alphas_cumprod(),
        cfg=cfg,
        device=torch.device("cpu"),
        generator=generator,
        n_noisy=2,
    )
    assert len(captured) == len(get_levels())
    return [
        (rows, out.eigenvalues[index], out.eigenvectors[index], False)
        for index, rows in enumerate(captured)
    ]


# one case per matrix that the notebooks plot the eigenvectors of; those of the newer
# estimators are skipped until the multigeom installed has them
CASES = [
    pytest.param(get_j_case, id="J"),
    pytest.param(get_g_case, id="G"),
    pytest.param(
        get_m_t_case,
        id="M_t",
        marks=pytest.mark.skipif(
            compute_g_per_scale is None,
            reason="needs a multigeom with compute_g_per_scale",
        ),
    ),
    pytest.param(
        get_env_case,
        id="M_t_env",
        marks=pytest.mark.skipif(
            compute_env_per_scale is None,
            reason="needs a multigeom with compute_env_per_scale",
        ),
    ),
]


@pytest.mark.parametrize("get_case", CASES)
def test_eigenvectors_are_eigenvectors(get_case, monkeypatch):
    """
    Verify that every eigenvector that the notebooks plot is an eigenvector of its
    matrix, with the right eigenvalue: for the Fisher information of the responses
    J(x) (all its pairs), the multiscale Fisher information G(x), G(x) at each noise
    level, M_t(x), and the stimulus features at each noise level, M_t^env(x) (the top
    10 pairs, at every noise level), each pair of eigenvalue and eigenvector (λ_i,
    v_i) has a relative residual ‖M v_i − λ_i v_i‖ / ‖M‖ below 1e-5 and a unit
    eigenvector. M v_i is computed as Dᵀ(D v_i) from the rows D of the same samples
    that the estimator drew, without building M.
    """
    # each matrix (one per noise level for the estimates at each level)
    for rows, eigenvalues, eigenvectors, squared in get_case(monkeypatch):
        assert_eigenpairs(rows, eigenvalues, eigenvectors, squared)


def test_the_check_fails_on_mismatched_eigenpairs():
    """
    Verify that eigenpairs that are not those of DᵀD fail the residual check,
    so that the test above cannot pass by accident: the eigenvalues paired with the
    next eigenvector (a shifted index), eigenvectors whose 8 x 8 images are transposed
    (a wrong reshape), and an eigenvector scaled by 2 (not a unit vector).
    """
    # random rows, and the exact top eigenpairs of DᵀD from their SVD
    rows = torch.randn(30, 64, generator=torch.Generator().manual_seed(4))
    _, singular_values, right = torch.linalg.svd(rows, full_matrices=False)
    values, vectors = singular_values[:TOP_K] ** 2, right[:TOP_K]

    # the exact pairs pass
    assert_eigenpairs(rows, values, vectors)

    # a shifted index, and a wrong reshape (each eigenvector's image transposed), fail
    transposed = vectors.view(TOP_K, 8, 8).transpose(1, 2).reshape(TOP_K, 64)
    for wrong in (vectors.roll(1, dims=0), transposed):
        residuals, _ = get_residuals(rows, values, wrong)
        assert float(residuals.max()) > 1e-2

    # an eigenvector that is not a unit vector fails
    with pytest.raises(AssertionError):
        assert_eigenpairs(rows, values, vectors * 2)
