"""
Mock models and settings shared by the unit tests of the estimators of G(x) and of
G(x) at each noise level (`compute_g`, `compute_g_per_scale`): a small image, a mock
encoder, mock diffusion models (one of which records its calls), and the differences of
the denoised images that the estimator drew, rebuilt from those calls.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import torch
import torch.nn as nn
from multigeom.config import load_config


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


def run_estimator(estimate, denoiser: nn.Module, top_k: int = PARAMS["top_k"]):
    """
    `estimate` (`compute_g` or `compute_g_per_scale`) for a random image, the mock
    encoder and the mock `denoiser` (with its noise schedule), keeping `top_k`
    eigenpairs.
    """
    # the noise schedule of the denoiser, and a random image in [-1, 1]
    alphas_cumprod = denoiser.alphas_cumprod
    size = PARAMS["image_size"]
    generator = torch.Generator().manual_seed(0)
    image = torch.rand(1, 1, size, size, generator=generator) * 2 - 1

    # the settings, with top_k eigenpairs kept
    cfg = get_cfg()
    cfg.top_k = top_k

    # the estimate, on the CPU
    return estimate(
        image=image,
        encoder=MockEncoder(),
        unet=denoiser,
        alphas_cumprod=alphas_cumprod,
        cfg=cfg,
        device=torch.device("cpu"),
        generator=generator,
    )
