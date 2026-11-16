"""
Unit tests of `multigeom.compute_g_per_scale` (notebook 7) and of the differences of
noise predictions it is built from, with mock models:

  - the integrand M_t(x) of G(x) vanishes at every noise level when the denoiser
    ignores the responses r, and does not when it reads them (a control, so that the
    zero is not vacuous). If the denoiser ignores r, the two responses r, r' of each
    sample give the same noise prediction, so every row of D, ε(x_t, t, r) −
    ε(x_t, t, r'), is zero: the responses carry no information about the image, and
    M_t(x) = 0 exactly;
  - every difference ε(x_t, t, r) − ε(x_t, t, r') is zero when the two responses are
    identical, r = r' (no observation noise, σ = 0), even for a denoiser that reads r,
    and not with the default σ (the control);
  - the trace of the estimate at each noise level is tr M̂_t = Σ_s |δ_s|² / (2 N t),
    with δ_s = x̂(x_t, r) − x̂(x_t, r') the difference of the two denoised images of
    sample s (in image units), N the number of samples and t the noise variance;
  - the explicit matrix M̂_t = Σ_s δ_s δ_sᵀ / (2 N t) is symmetric and positive
    semi-definite within numerical tolerance, and so is the matrix rebuilt from the
    eigenpairs that compute_g_per_scale returns, which is the same matrix;
  - the curve tr M̂_t equals t · trace(⟨J_t⟩), with J_t the Fisher information of the
    responses about the noisy image, for a linear-Gaussian model whose posterior means
    are exact, when the responses are drawn with the spread they have given the noisy
    image. The estimator draws them given the denoised image instead, an approximation
    that lowers the curve.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

import multigeom
from multigeom.config import load_config
from multigeom.geometry.methods import ConditionalPair

# compute_g_per_scale is newer than the multigeom on PyPI: its tests are skipped until
# the installed one has it
compute_g_per_scale = getattr(multigeom, "compute_g_per_scale", None)
needs_per_scale = pytest.mark.skipif(
    compute_g_per_scale is None, reason="needs a multigeom with compute_g_per_scale"
)

# the settings of the tests: a small image, a few neurons, five noise levels (DDPM steps
# 100 to 900, every 200) with 20 samples each, and three eigenpairs per level
PARAMS = {
    "image_size": 8,
    "n_neurons": 3,
    "n_samples": 20,
    "batch_size": 10,
    "grid": (100, 900, 200),
    "top_k": 3,
    # the DDPM steps of the check against the Fisher information, from fine to coarse
    "fisher_levels": (10, 100, 300, 600, 900),
}


class ResponseBlindDenoiser(nn.Module):
    """
    A mock diffusion model whose noise prediction depends on the noisy image and the
    noise level, but never on the response (the null token included).
    """

    def __init__(self, alphas_cumprod: torch.Tensor):
        super().__init__()

        # ᾱ of each DDPM step, to make the prediction depend on the noise level
        self.register_buffer("alphas_cumprod", alphas_cumprod)

    def forward(
        self, x_t: torch.Tensor, timestep: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        # a function of the noisy image and the noise level only: `response` is not read
        scale = self.alphas_cumprod[timestep].sqrt().view(-1, 1, 1, 1)
        return torch.tanh(x_t) * scale


class ResponseReadingDenoiser(ResponseBlindDenoiser):
    """The same mock plus a term that reads the response: the control."""

    def forward(
        self, x_t: torch.Tensor, timestep: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        # the sum of the responses, added to every pixel
        readout = response.sum(dim=1).view(-1, 1, 1, 1) * torch.ones_like(x_t)
        return super().forward(x_t, timestep, response) + readout


class RecordingDenoiser(ResponseBlindDenoiser):
    """
    A mock that reads the response through a fixed random map to the pixels, modulated
    by the noisy image (so that the difference of the two predictions changes from
    sample to sample and M_t has many non-zero eigenvalues), and records every call:
    the noisy images, the DDPM step, the responses and the noise prediction.
    """

    def __init__(self, alphas_cumprod: torch.Tensor):
        super().__init__(alphas_cumprod)

        # the fixed map from the responses to the pixels [P, n_neurons]
        n_pixels = PARAMS["image_size"] ** 2
        generator = torch.Generator().manual_seed(1)
        readout = torch.randn(n_pixels, PARAMS["n_neurons"], generator=generator)
        self.register_buffer("readout", readout)

        # the calls, in order
        self.calls = []

    def forward(
        self, x_t: torch.Tensor, timestep: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        # the responses mapped to the pixels, modulated by the noisy images
        mapped = (response @ self.readout.T).view_as(x_t) * (1 + 0.5 * torch.tanh(x_t))
        prediction = super().forward(x_t, timestep, response) + mapped

        # a copy of the inputs and of the output of this call
        self.calls.append(
            {
                "x_t": x_t.clone(),
                "timestep": timestep.clone(),
                "response": response.clone(),
                "prediction": prediction.clone(),
            }
        )
        return prediction


class MockEncoder(nn.Module):
    """A mock encoder: responses [B, n_neurons] that depend on the image."""

    n_neurons = PARAMS["n_neurons"]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # the mean of each image, with a different gain for each neuron
        gains = torch.arange(1.0, self.n_neurons + 1)
        return images.flatten(1).mean(dim=1, keepdim=True) * gains


class GaussianDenoiser(nn.Module):
    """
    The exact posterior means of a linear-Gaussian model, as the noise prediction ε of
    a diffusion network: prior x ~ N(0, C), responses r = A x + σ ξ, noisy image
    y = x + √t ε (variance-exploding). The null token gives E[x | y]; a response gives
    E[x | y, r].
    """

    def __init__(self, alphas_cumprod: torch.Tensor, model: dict):
        super().__init__()

        # ᾱ of each DDPM step, the prior precision C⁻¹, the readout A and σ, in float64
        self.register_buffer("alphas_cumprod", alphas_cumprod.double())
        self.register_buffer("prior_precision", torch.linalg.inv(model["prior"]))
        self.register_buffer("readout", model["readout"])
        self.sigma = model["sigma"]

    def forward(
        self, x_t: torch.Tensor, timestep: torch.Tensor, response: torch.Tensor
    ) -> torch.Tensor:
        # the noise variance t of each image, and the image in variance-exploding units
        alpha_bar = self.alphas_cumprod[timestep].view(-1, 1)
        noise_variance = (1 - alpha_bar) / alpha_bar
        y = x_t.flatten(1).double() / alpha_bar.sqrt()

        # the posterior precision: the prior and the noisy image, plus the response
        # unless it is the null token (all zeros)
        has_response = response.any(dim=1).double().view(-1, 1)
        likelihood = self.readout.T @ self.readout / self.sigma**2
        identity = torch.eye(y.shape[1], dtype=torch.float64)
        precision = self.prior_precision + identity / noise_variance.view(-1, 1, 1)
        precision = precision + has_response.view(-1, 1, 1) * likelihood

        # the evidence of the noisy image and of the response, weighted by precision
        readout_evidence = response.double() @ self.readout / self.sigma**2
        evidence = y / noise_variance + has_response * readout_evidence

        # the posterior mean x̂, one linear system per image
        x_hat = torch.linalg.solve(precision, evidence.unsqueeze(-1)).squeeze(-1)

        # the noise prediction that Tweedie's formula turns back into x̂
        noisy = x_t.flatten(1).double()
        eps = (noisy - alpha_bar.sqrt() * x_hat) / (1 - alpha_bar).sqrt()
        return eps.view_as(x_t).to(x_t.dtype)


class LinearEncoder(nn.Module):
    """The encoder of the linear-Gaussian model: the responses A x, without noise."""

    def __init__(self, readout: torch.Tensor):
        super().__init__()

        # the readout A [n, P], and its number of neurons
        self.register_buffer("readout", readout.float())
        self.n_neurons = readout.shape[0]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # A x for each image
        return images.flatten(1) @ self.readout.T


def get_gaussian_model() -> dict:
    """
    A linear-Gaussian model on images of PARAMS["image_size"]² pixels, in float64: the
    prior covariance C = U diag(c) Uᵀ, whose first n_neurons eigen-directions have the
    variance 1 and the others between 0.5 and 0.001; the readout A, which mixes those
    first directions, so that given the noisy image the responses spread the same way
    in every direction; and the response noise σ.
    """
    # a random orthogonal basis U, and the prior variances c
    n_pixels, n_neurons = PARAMS["image_size"] ** 2, PARAMS["n_neurons"]
    generator = torch.Generator().manual_seed(2)
    draw = {"generator": generator, "dtype": torch.float64}
    basis = torch.linalg.qr(torch.randn(n_pixels, n_pixels, **draw))[0]
    others = 0.5 * 0.9 ** torch.arange(n_pixels - n_neurons, dtype=torch.float64)
    variances = torch.cat([torch.ones(n_neurons, dtype=torch.float64), others])

    # the readout: a random rotation of the first n_neurons directions
    rotation = torch.linalg.qr(torch.randn(n_neurons, n_neurons, **draw))[0]
    readout = rotation @ basis[:, :n_neurons].T
    prior = basis @ torch.diag(variances) @ basis.T
    return {"prior": prior, "readout": readout, "sigma": 0.3}


def get_fisher_curve(model: dict, noise_variance: float) -> tuple[float, float]:
    """
    For the linear-Gaussian model at the noise variance t: t · trace(J_t), with
    J_t = K_tᵀ Aᵀ S_t⁻¹ A K_t the Fisher information of the responses about the noisy
    image (K_t = C (C + t I)⁻¹, S_t = A Σ_t Aᵀ + σ² I with Σ_t the posterior covariance
    of the image), and the spread of the responses given the noisy image, S_t[0, 0]
    (S_t is that times the identity in this model).
    """
    # the posterior mean gain K_t and covariance Σ_t of the image given the noisy image
    prior, readout, t = model["prior"], model["readout"], noise_variance
    identity = torch.eye(len(prior), dtype=torch.float64)
    gain = prior @ torch.linalg.inv(prior + t * identity)
    posterior = torch.linalg.inv(torch.linalg.inv(prior) + identity / t)

    # the covariance of the responses given the noisy image, the same in every direction
    spread = readout @ posterior @ readout.T + model["sigma"] ** 2 * torch.eye(
        len(readout), dtype=torch.float64
    )
    torch.testing.assert_close(spread, spread[0, 0] * torch.eye(len(readout)).double())

    # the Fisher information of the responses about the noisy image, times t
    fisher = gain.T @ readout.T @ torch.linalg.inv(spread) @ readout @ gain
    return t * float(torch.trace(fisher)), float(spread[0, 0])


def get_alphas_cumprod() -> torch.Tensor:
    """ᾱ_t of the paper's linear β schedule (1000 DDPM steps, β from 1e-4 to 0.02)."""
    # β_t, then ᾱ_t = Π_{s≤t} (1 − β_s)
    betas = torch.linspace(1e-4, 0.02, 1000)
    return torch.cumprod(1.0 - betas, dim=0)


def get_cfg() -> SimpleNamespace:
    """The packaged estimator settings, with the small sizes of PARAMS."""
    # a copy of the packaged settings (σ of the responses, seed, ...)
    cfg = copy.deepcopy(load_config("estimator"))

    # fewer samples, a coarser grid of noise levels, fewer eigenpairs
    cfg.monte_carlo.n_samples = PARAMS["n_samples"]
    cfg.monte_carlo.batch_size = PARAMS["batch_size"]
    grid = cfg.monte_carlo.timestep_grid
    grid.start, grid.stop, grid.step = PARAMS["grid"]
    cfg.top_k = PARAMS["top_k"]
    return cfg


def get_levels() -> list[int]:
    """The DDPM steps of the noise levels of PARAMS, stop included."""
    # start, stop and step of the grid
    start, stop, step = PARAMS["grid"]
    return list(range(start, stop + 1, step))


def compute_differences(sigma: float) -> list[torch.Tensor]:
    """
    The rows of D, ε(x_t, t, r) − ε(x_t, t, r'), that the conditional-pair method draws
    from one batch of noisy images at each noise level, for a random image, the mock
    encoder and the mock denoiser that reads the response, when the responses have
    observation noise `sigma` (σ = 0: r = r').
    """
    # the noise schedule, the settings with this observation noise, and a random image
    alphas_cumprod = get_alphas_cumprod()
    cfg = get_cfg()
    cfg.monte_carlo.observation_noise_sigma = sigma
    size, batch = PARAMS["image_size"], PARAMS["batch_size"]
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(1, 1, size, size, generator=generator) * 2 - 1

    # the method that draws the rows of D (lines 7-14 of Algorithm 1)
    sampler = ConditionalPair(
        image=image,
        encoder=MockEncoder(),
        model=ResponseReadingDenoiser(alphas_cumprod),
        alphas_cumprod=alphas_cumprod,
        cfg=cfg,
        device=torch.device("cpu"),
        generator=generator,
    )

    # at each noise level: a batch of noisy images x_t = √ᾱ x + √(1 − ᾱ) z, and its rows
    differences = []
    for level in get_levels():
        noise = torch.randn(batch, 1, size, size, generator=generator)
        weights = alphas_cumprod[level].sqrt(), (1 - alphas_cumprod[level]).sqrt()
        x_t = weights[0] * image + weights[1] * noise
        differences.append(sampler.sample_rows(x_t, torch.full((batch,), level), level))
    return differences


def compute_m_t(denoiser: nn.Module, top_k: int = PARAMS["top_k"]) -> SimpleNamespace:
    """
    The top `top_k` eigenpairs of M_t(x) at each noise level, for a random image, the
    mock encoder and the mock `denoiser` (with its noise schedule).
    """
    # the noise schedule of the denoiser, and a random image in [-1, 1]
    alphas_cumprod = denoiser.alphas_cumprod
    size = PARAMS["image_size"]
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(1, 1, size, size, generator=generator) * 2 - 1

    # the settings, with top_k eigenpairs kept at each noise level
    cfg = get_cfg()
    cfg.top_k = top_k

    # M_t(x) at each noise level, on the CPU
    return compute_g_per_scale(
        image=image,
        encoder=MockEncoder(),
        unet=denoiser,
        alphas_cumprod=alphas_cumprod,
        cfg=cfg,
        device=torch.device("cpu"),
        generator=generator,
    )


def compute_deltas(calls: list[dict], alphas_cumprod: torch.Tensor) -> dict:
    """
    The differences δ_s = x̂(x_t, r) − x̂(x_t, r') of the two denoised images of each
    sample s (Tweedie's formula, in image units, float64) at each noise level, from the
    calls that the estimator made to a RecordingDenoiser: {DDPM step: [N, P]}.
    """
    # the calls with a response (the null token is all zeros): the two of each sample,
    # r then r', follow each other
    conditional = [call for call in calls if bool(call["response"].any())]
    deltas = {}
    for first, second in zip(conditional[0::2], conditional[1::2]):
        # the two calls of a pair are on the same noisy images x_t
        assert torch.equal(first["x_t"], second["x_t"])

        # the denoised images x̂ = (x_t − √(1 − ᾱ) ε) / √ᾱ for r and r', in float64
        level = int(first["timestep"][0])
        alpha_bar = alphas_cumprod[level].double()
        x_hats = [
            (
                call["x_t"].double()
                - (1 - alpha_bar).sqrt() * call["prediction"].double()
            )
            / alpha_bar.sqrt()
            for call in (first, second)
        ]

        # δ_s for each sample of the batch, after those of the earlier batches
        batch = (x_hats[0] - x_hats[1]).flatten(1)
        deltas[level] = torch.cat([deltas[level], batch]) if level in deltas else batch
    return deltas


def get_noise_variance(alphas_cumprod: torch.Tensor, level: int) -> float:
    """t = (1 − ᾱ) / ᾱ, the noise variance of a DDPM step, in image units."""
    # ᾱ of the step, in float64
    alpha_bar = float(alphas_cumprod[level].double())
    return (1 - alpha_bar) / alpha_bar


def compute_expected_traces(
    calls: list[dict], alphas_cumprod: torch.Tensor
) -> tuple[dict, dict]:
    """
    The expected trace of M̂_t at each noise level, Σ_s |δ_s|² / (2 N t), from the calls
    that the estimator made to a RecordingDenoiser (see `compute_deltas`), with N the
    number of samples of the level and t its noise variance. Returns ({DDPM step:
    expected trace}, {DDPM step: N}).
    """
    # δ_s of every sample, at each level
    deltas = compute_deltas(calls, alphas_cumprod)

    # Σ_s |δ_s|² / (2 N t), and N, at each level
    traces, counts = {}, {}
    for level, rows in deltas.items():
        noise_variance = get_noise_variance(alphas_cumprod, level)
        traces[level] = float(rows.pow(2).sum()) / (2 * len(rows) * noise_variance)
        counts[level] = len(rows)
    return traces, counts


def compute_explicit_matrices(calls: list[dict], alphas_cumprod: torch.Tensor) -> dict:
    """
    The explicit matrix M̂_t = Σ_s δ_s δ_sᵀ / (2 N t) [P, P] at each noise level, in
    float64, from the calls that the estimator made to a RecordingDenoiser (see
    `compute_deltas`): {DDPM step: matrix}.
    """
    # δ_s of every sample, at each level
    deltas = compute_deltas(calls, alphas_cumprod)

    # the sum of the outer products δ_s δ_sᵀ, over 2 N t, at each level
    matrices = {}
    for level, rows in deltas.items():
        noise_variance = get_noise_variance(alphas_cumprod, level)
        matrices[level] = rows.T @ rows / (2 * len(rows) * noise_variance)
    return matrices


@needs_per_scale
def test_m_t_is_zero_when_the_denoiser_ignores_the_response():
    out = compute_m_t(ResponseBlindDenoiser(get_alphas_cumprod()))

    # three eigenvalues at each of the five noise levels, all exactly zero
    assert tuple(out.eigenvalues.shape) == (len(get_levels()), PARAMS["top_k"])
    assert torch.equal(out.eigenvalues, torch.zeros_like(out.eigenvalues))


@needs_per_scale
def test_m_t_is_not_zero_when_the_denoiser_reads_the_response():
    out = compute_m_t(ResponseReadingDenoiser(get_alphas_cumprod()))

    # the leading eigenvalue is positive at every noise level
    assert bool((out.eigenvalues[:, 0] > 0).all()), out.eigenvalues[:, 0]


def test_every_difference_is_zero_when_the_responses_are_identical():
    # σ = 0: the two responses of each sample are the same, r = r' = f(x̂(x_t))
    differences = compute_differences(sigma=0.0)

    # one batch of rows [B, P] at each of the five noise levels, every one exactly zero
    size, batch = PARAMS["image_size"], PARAMS["batch_size"]
    assert len(differences) == len(get_levels())
    for rows in differences:
        assert tuple(rows.shape) == (batch, size * size)
        assert torch.equal(rows, torch.zeros_like(rows))


def test_differences_are_not_zero_when_the_responses_differ():
    # the control: with the default σ, r and r' differ, and the denoiser reads them
    sigma = float(load_config("estimator").monte_carlo.observation_noise_sigma)

    # every row is non-zero, at every noise level
    for rows in compute_differences(sigma=sigma):
        assert bool((rows.abs().sum(dim=1) > 0).all())


@needs_per_scale
def test_trace_of_m_t_is_the_sum_of_squared_differences_over_2nt():
    """
    tr M̂_t = Σ_s |δ_s|² / (2 N t) at every noise level: the sum of all the eigenvalues
    of the estimate equals that of the squared differences of the two denoised images
    of each sample, over 2 N t. M̂_t has rank at most N, so N eigenpairs are all of it.
    """
    # M̂_t with all its N eigenvalues, with a denoiser that records its calls
    denoiser = RecordingDenoiser(get_alphas_cumprod())
    out = compute_m_t(denoiser, top_k=PARAMS["n_samples"])

    # the expected traces, from the noisy images, responses and predictions it used
    expected, counts = compute_expected_traces(denoiser.calls, denoiser.alphas_cumprod)

    # N samples at each of the five levels, and several non-zero eigenvalues per level
    levels = get_levels()
    assert counts == {level: PARAMS["n_samples"] for level in levels}
    assert bool((out.eigenvalues[:, 2] > 1e-6 * out.eigenvalues[:, 0]).all())

    # the trace of M̂_t at each level, against Σ_s |δ_s|² / (2 N t)
    traces = out.eigenvalues.double().sum(dim=1)
    wanted = torch.tensor([expected[level] for level in levels], dtype=torch.float64)
    torch.testing.assert_close(traces, wanted, rtol=1e-4, atol=0)


@needs_per_scale
def test_explicit_matrix_is_symmetric_and_positive_semidefinite():
    """
    Explicit matrix check: at every noise level, the whole P x P matrix
    M̂_t = Σ_s δ_s δ_sᵀ / (2 N t), built from the differences of the denoised images that
    the estimator drew, is symmetric and positive semi-definite within numerical
    tolerance; so is the matrix rebuilt from the N eigenpairs that compute_g_per_scale
    returns, and the two are the same matrix.
    """
    # M̂_t with all its N eigenpairs, with a denoiser that records its calls
    denoiser = RecordingDenoiser(get_alphas_cumprod())
    out = compute_m_t(denoiser, top_k=PARAMS["n_samples"])

    # the explicit matrices, from the noisy images, responses and predictions it used
    explicit = compute_explicit_matrices(denoiser.calls, denoiser.alphas_cumprod)
    for index, level in enumerate(get_levels()):
        # the matrix rebuilt from the eigenpairs of the estimate, Vᵀ diag(λ) V
        vectors = out.eigenvectors[index].double()
        rebuilt = vectors.T @ torch.diag(out.eigenvalues[index].double()) @ vectors

        for matrix in (explicit[level], rebuilt):
            # symmetric, within rounding
            tolerance = 1e-12 * float(matrix.abs().max())
            torch.testing.assert_close(matrix, matrix.T, rtol=0, atol=tolerance)

            # positive semi-definite: no eigenvalue below -1e-10 times the largest
            eigenvalues = torch.linalg.eigvalsh(matrix)
            assert float(eigenvalues.min()) >= -1e-10 * float(eigenvalues.max())

        # the eigenpairs of the estimate describe the explicit matrix
        error = torch.linalg.matrix_norm(rebuilt - explicit[level])
        assert float(error / torch.linalg.matrix_norm(explicit[level])) < 1e-5


def compute_trace_at(level: int, model: dict, spread: float) -> float:
    """
    The trace of the estimate of M_t(x) at the DDPM step `level`, for a random image and
    the linear-Gaussian `model`, with the responses drawn with the variance `spread`
    around f(x̂(x_t)): the sum of all its eigenvalues (its rank is at most n_neurons,
    fewer than the eigenpairs kept).
    """
    # the settings: this noise level only, many samples, the spread of the responses
    cfg = get_cfg()
    grid = cfg.monte_carlo.timestep_grid
    grid.start, grid.stop, grid.step = level, level, 1
    cfg.monte_carlo.n_samples, cfg.monte_carlo.batch_size = 4000, 1000
    cfg.monte_carlo.observation_noise_sigma = spread**0.5
    cfg.top_k = PARAMS["n_neurons"] + 2

    # a random image, and the estimate with the exact posterior means of the model
    alphas_cumprod = get_alphas_cumprod()
    size = PARAMS["image_size"]
    generator = torch.Generator().manual_seed(level)
    image = torch.rand(1, 1, size, size, generator=generator) * 2 - 1
    out = compute_g_per_scale(
        image=image,
        encoder=LinearEncoder(model["readout"]),
        unet=GaussianDenoiser(alphas_cumprod, model),
        alphas_cumprod=alphas_cumprod,
        cfg=cfg,
        device=torch.device("cpu"),
        generator=generator,
    )
    return float(out.eigenvalues.double().sum())


@needs_per_scale
def test_the_curve_equals_t_times_the_trace_of_the_fisher_information():
    """
    The curve tr M̂_t equals t · trace(⟨J_t⟩) at noise levels from fine to coarse, J_t
    the Fisher information of the responses about the noisy image, for the
    linear-Gaussian model (where J_t does not depend on the noisy image, so
    ⟨J_t⟩ = J_t). The estimator draws the responses given the denoised image,
    N(f(x̂), σ²); here they are drawn with the spread they have given the noisy image,
    S_t = A Σ_t Aᵀ + σ² I (the same in every direction in this model), so that the
    samples are exact. Monte-Carlo error about 2% with 4000 samples; tolerance 10%.
    """
    model = get_gaussian_model()
    alphas_cumprod = get_alphas_cumprod()
    for level in PARAMS["fisher_levels"]:
        # t · trace(J_t), and the spread of the responses, in closed form
        noise_variance = get_noise_variance(alphas_cumprod, level)
        expected, spread = get_fisher_curve(model, noise_variance)

        # the trace of the estimate, with exact samples of the responses
        trace = compute_trace_at(level, model, spread)
        assert trace == pytest.approx(expected, rel=0.1), (level, trace, expected)
