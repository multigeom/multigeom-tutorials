"""
Helpers of notebooks 7 and 8, on the scale-resolved decomposition of G(x): the example
images, the estimates at each noise level, and the figures of their leading
eigenvectors.

    load_examples       the frog and the test images of the landing page, with seeds
    compute_per_scale   M_t(x), the integrand of G(x), at each level (notebooks 7, 8)
    compute_env         M_t^env(x), the stimulus features, at each level (notebook 8)
    align_signs         the signs of the eigenvectors, for display
    plot_per_scale      the figure of one image: each block's eigenvectors per level
    compute_summary     how the leading eigenvector depends on the image (notebook 7)
    plot_summary
    compute_alignment   the alignment of M_t(x) with M_t^env(x) per level (notebook 8)
    plot_alignment

The functions of the package are looked up when they are called (`multigeom.<name>`), so
that this module imports with a multigeom that does not have them yet.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from matplotlib.gridspec import GridSpec
from PIL import Image

import multigeom
from multigeom.analysis import subspace_alignment
from multigeom.datasets import CANONICAL_TRANSFORM, TinyImageNetDataset
from multigeom.utils import make_generator

from .style import AREA_COLORS, panel_label, plt, show_eigenvector

# the colour of each block of eigenvectors: the areas, and the stimulus features
BLOCK_COLORS = {**AREA_COLORS, "stimulus": "#4D4D4D"}

# the layout of the figure of one image: the height of its rows (in units of a tile),
# its width, the height with two blocks, and the extra height of each further block
EIGVEC_ROW, BLOCK_GAP, VALUES_GAP, VALUES_ROW = 1.0, 0.45, 0.9, 3.2
FIGURE_WIDTH, TWO_BLOCK_HEIGHT, BLOCK_HEIGHT = 16.5, 8.6, 3.05


def load_examples(params: dict, seed: int) -> dict:
    """
    The example images, [1, 1, 64, 64] in [-1, 1], each with the seed of its
    Monte-Carlo draws: {name: (image, seed)}. The frog has the seed of the estimator
    config, as in notebook 3; test image i has that seed + i, as in the pipeline of
    notebook 4.
    """
    # the frog, a training image, read from its file
    frog = CANONICAL_TRANSFORM(Image.open(params["frog"])).unsqueeze(0)
    examples = {"frog": (frog, seed)}

    # the test images (validation images), by their index
    test_set = TinyImageNetDataset(split="val", transform=CANONICAL_TRANSFORM)
    for index in params["test_images"]:
        examples[f"img{index:04d}"] = (test_set[index].unsqueeze(0), seed + index)
    return examples


def compute_per_scale(
    examples: dict,
    encoder: torch.nn.Module,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
) -> dict:
    """
    The leading eigenpairs of M_t(x) at each noise level, for every example image, with
    one encoder and its diffusion model: {name: PerScaleOutput}.
    """
    # one estimate per image, with its own seeded random draws
    return {
        name: multigeom.compute_g_per_scale(
            image=image,
            encoder=encoder,
            unet=unet,
            alphas_cumprod=alphas_cumprod,
            cfg=cfg,
            device=device,
            generator=make_generator(seed, device),
        )
        for name, (image, seed) in examples.items()
    }


def compute_env(
    examples: dict,
    unet: torch.nn.Module,
    alphas_cumprod: torch.Tensor,
    cfg: SimpleNamespace,
    device: torch.device,
    params: dict,
) -> dict:
    """
    The leading eigenpairs of the stimulus features M_t^env(x) at each noise level, for
    every example image, from the null token of the diffusion model `unet`, with the
    `n_noisy` noisy images per probe and the step `eta` of `params`: {name:
    PerScaleOutput}. The seed of each image is shifted by `env_seed_offset`, so that
    its random draws are independent of those of M_t(x) (`compute_per_scale`).
    """
    # one estimate per image, with its own seeded random draws
    offset = params["env_seed_offset"]
    return {
        name: multigeom.compute_env_per_scale(
            image=image,
            unet=unet,
            alphas_cumprod=alphas_cumprod,
            cfg=cfg,
            device=device,
            generator=make_generator(seed + offset, device),
            n_noisy=params["n_noisy"],
            eta=params["eta"],
        )
        for name, (image, seed) in examples.items()
    }


def align_signs(eigvecs: torch.Tensor) -> torch.Tensor:
    """
    The eigenvectors [J, k, P] of the J noise levels, each with the sign that agrees
    with the same eigenvector at the next coarser level (for display: the sign of an
    eigenvector is arbitrary); those of the coarsest level have a positive sum.
    """
    # a copy, to leave the results untouched
    aligned = eigvecs.clone()

    # the coarsest level (the last one): a positive sum over the pixels
    aligned[-1] *= torch.where(aligned[-1].sum(dim=1, keepdim=True) < 0, -1.0, 1.0)

    # then level by level towards the finest: the sign of the coarser neighbour
    for level in range(aligned.shape[0] - 2, -1, -1):
        overlap = (aligned[level] * aligned[level + 1]).sum(dim=1, keepdim=True)
        aligned[level] *= torch.where(overlap < 0, -1.0, 1.0)
    return aligned


def get_plot_data(image: torch.Tensor, records: dict) -> dict:
    """
    The data of the figure of one image: the image [64, 64], tau = log t [J], and for
    each block its eigenvalues [J, k] and its eigenvectors [J, k, P] with the signs of
    `align_signs`. `records` maps each block (an area, or "stimulus") to its
    PerScaleOutput.
    """
    # the image, and the noise levels (the same for every block)
    first = next(iter(records.values()))
    data = {"image": image.reshape(64, 64).numpy(), "log_t": first.log_tau.numpy()}

    # the eigenvalues and the signed eigenvectors of each block
    for name, record in records.items():
        data[name] = {
            "eigenvalues": record.eigenvalues.numpy(),
            "eigenvectors": align_signs(record.eigenvectors).numpy(),
        }
    return data


def draw_block(
    fig: plt.Figure, grid: GridSpec, top: int, data: dict, name: str, header: bool
) -> plt.Axes:
    """
    The eigenvectors of one block, from row `top` of `grid` (the tiles start at its
    fourth column): one row per eigenvector, one column per noise level, with the
    noise levels above the columns if `header`. Returns the top-left tile.
    """
    # the eigenvectors [J, k, P] of the block: J noise levels, k eigenvectors each
    eigvecs = data[name]["eigenvectors"]
    n_levels, n_eigvecs = eigvecs.shape[:2]

    # one tile per eigenvector (row) and noise level (column), each eigenvector drawn
    tiles = [
        [fig.add_subplot(grid[top + row, 3 + level]) for level in range(n_levels)]
        for row in range(n_eigvecs)
    ]
    for row in range(n_eigvecs):
        for level in range(n_levels):
            show_eigenvector(tiles[row][level], eigvecs[level, row])

    # the names of the rows (v1, v2, ...) in the first column
    for row in range(n_eigvecs):
        tiles[row][0].set_ylabel(
            f"$v_{row + 1}$", fontsize=9, rotation=0, ha="right", va="center"
        )

    # the noise level tau = log t above each column, if asked
    if header:
        for level in range(n_levels):
            label = f"{data['log_t'][level]:.1f}".replace("-", "−")
            tiles[0][level].set_title(label, fontsize=7)

    # the name of the block, left of the names of its rows
    top_left, bottom_left = tiles[0][0].get_position(), tiles[-1][0].get_position()
    middle = (top_left.y1 + bottom_left.y0) / 2
    fig.text(
        top_left.x0 - 0.028,
        middle,
        name,
        fontsize=12,
        rotation=90,
        ha="center",
        va="center",
    )
    return tiles[0][0]


def draw_eigenvalues(ax: plt.Axes, data: dict, names: list) -> None:
    """
    The eigenvalues of each block against tau = log t, the leading one thick and the
    others thin, in the columns of the tiles (one per noise level).
    """
    # one line per eigenvalue and block
    n_levels, n_eigvecs = data[names[0]]["eigenvalues"].shape
    for name in names:
        for k in range(n_eigvecs):
            ax.plot(
                np.arange(n_levels),
                data[name]["eigenvalues"][:, k],
                color=BLOCK_COLORS[name],
                lw=2.0 if k == 0 else 0.8,
                alpha=1.0 if k == 0 else 0.6,
                label=name if k == 0 else None,
            )

    # the columns of the tiles, labelled with tau = log t; a log scale for the values
    labels = [f"{value:.1f}".replace("-", "−") for value in data["log_t"]]
    ax.set_xlim(-0.5, n_levels - 0.5)
    ax.set_xticks(np.arange(n_levels), labels, fontsize=7)
    ax.set_yscale("log")
    ax.set_xlabel(r"$\tau = \log t$   (t: variance of the noise, in image units)")
    matrices = (
        r"$M_t^{\rm env}(x)$ and $M_t(x)$" if "stimulus" in names else r"$M_t(x)$"
    )
    ax.set_ylabel("eigenvalues\nof " + matrices)

    # which line is which
    ax.legend(
        title=rf"thick: $\lambda_1$, thin: $\lambda_2$ to $\lambda_{n_eigvecs}$",
        fontsize=8,
        title_fontsize=8,
        frameon=False,
        loc="lower left",
    )


def plot_per_scale(
    image: torch.Tensor, records: dict, params: dict, title: str, suptitle: str
) -> tuple[plt.Figure, dict]:
    """
    The figure of one image: the image (A); for each block of `records` (an area, or
    "stimulus", mapped to its PerScaleOutput, in that order) its leading eigenvectors at
    each noise level, one row per eigenvector and one column per level (B, C, ...); and
    their eigenvalues against tau = log t (last panel). Returns the figure and the data
    it shows.
    """
    # the data drawn, its blocks, and the numbers of noise levels and eigenvectors
    data = get_plot_data(image, records)
    names = list(records)
    n_levels, n_eigvecs = data[names[0]]["eigenvalues"].shape

    # the grid: the image on the left; for each block one row per eigenvector and one
    # column per noise level, blocks apart by a gap; the eigenvalues at the bottom
    block_rows = [EIGVEC_ROW] * n_eigvecs
    heights = (block_rows + [BLOCK_GAP]) * (len(names) - 1) + block_rows
    heights = heights + [VALUES_GAP, VALUES_ROW]
    height = TWO_BLOCK_HEIGHT + BLOCK_HEIGHT * (len(names) - 2)
    fig = plt.figure(figsize=(FIGURE_WIDTH, height), dpi=params["screen_dpi"])
    grid = fig.add_gridspec(
        len(heights),
        3 + n_levels,
        height_ratios=heights,
        width_ratios=[1, 1, 1.6] + [1] * n_levels,
        hspace=0.08,
        wspace=0.06,
    )

    # A: the image
    ax_image = fig.add_subplot(grid[0:2, 0:2])
    ax_image.imshow(data["image"], cmap="gray", vmin=-1, vmax=1)
    ax_image.set_title(title, fontsize=10)
    ax_image.axis("off")

    # B, C, ...: the eigenvectors of each block, the first with the noise levels above
    first_tiles = [
        draw_block(fig, grid, block * (n_eigvecs + 1), data, name, header=block == 0)
        for block, name in enumerate(names)
    ]

    # last panel: the eigenvalues, in the columns of the tiles
    ax_values = fig.add_subplot(grid[-1, 3:])
    draw_eigenvalues(ax_values, data, names)

    # the panel letters, left of the names of the rows and blocks
    axes = [ax_image, *first_tiles, ax_values]
    for letter, ax in zip("ABCDEFG", axes):
        panel_label(fig, ax, letter, dx=-0.01 if letter == "A" else -0.04)

    # the title, on two lines
    fig.suptitle(suptitle, fontsize=11, y=0.99)
    return fig, data


def compute_summary(results: dict, areas: list) -> dict:
    """
    Per noise level, across the example images: for each area, the mean |cos| between
    the leading eigenvectors of two different images, and the leading eigenvalues
    (all, and their median); and the fraction of images whose leading eigenvalue is
    larger in the second area than in the first.
    """
    # the image names, and the noise levels
    names = list(results[areas[0]])
    summary = {"log_t": results[areas[0]][names[0]].log_tau}

    # every pair of two different images (in both orders)
    pairs = ~torch.eye(len(names), dtype=torch.bool)
    for area in areas:
        # the leading eigenvectors [images, J, P] and eigenvalues [images, J]
        vectors = torch.stack(
            [results[area][name].eigenvectors[:, 0] for name in names]
        )
        values = torch.stack([results[area][name].eigenvalues[:, 0] for name in names])

        # |cos| between two images at each level [J, images, images], mean over pairs
        cos = torch.einsum("ajp,bjp->jab", vectors, vectors).abs()
        summary[f"similarity_{area}"] = cos[:, pairs].mean(dim=1)

        # the leading eigenvalues, and their median over the images
        summary[f"values_{area}"] = values
        summary[f"median_{area}"] = values.median(dim=0).values

    # the fraction of images where the second area has the larger leading eigenvalue
    larger = summary[f"values_{areas[1]}"] > summary[f"values_{areas[0]}"]
    summary["second_larger"] = larger.float().mean(dim=0)
    return summary


def plot_summary(summary: dict, params: dict) -> tuple[plt.Figure, dict]:
    """
    The summary across the example images against tau = log t, for each area: the mean
    |cos| between the leading eigenvectors of two images (A), and the median leading
    eigenvalue (B). Returns the figure and the data it shows.
    """
    # the data drawn
    areas = params["areas"]
    data = {"log_t": summary["log_t"].numpy()}
    for area in areas:
        data[area] = {
            "similarity": summary[f"similarity_{area}"].numpy(),
            "median": summary[f"median_{area}"].numpy(),
        }

    # two panels, one line per area in each
    fig, (ax_cos, ax_values) = plt.subplots(
        1, 2, figsize=(11, 3.6), dpi=params["screen_dpi"]
    )
    for area in areas:
        style = {"color": AREA_COLORS[area], "marker": "o", "ms": 3, "label": area}
        ax_cos.plot(data["log_t"], data[area]["similarity"], **style)
        ax_values.plot(data["log_t"], data[area]["median"], **style)

    # A: how much the leading eigenvector depends on the image
    ax_cos.set_ylim(0, 1.02)
    ax_cos.set_xlabel(r"$\tau = \log t$")
    ax_cos.set_ylabel("mean |cos| between the leading\neigenvectors of two images")
    ax_cos.legend(frameon=False)

    # B: the leading eigenvalue
    ax_values.set_yscale("log")
    ax_values.set_xlabel(r"$\tau = \log t$")
    ax_values.set_ylabel("leading eigenvalue of $M_t(x)$\n(median over the images)")

    # the panel letters, and the title
    for letter, ax in zip("AB", (ax_cos, ax_values)):
        panel_label(fig, ax, letter)
    n_images = summary[f"values_{areas[0]}"].shape[0]
    fig.suptitle(
        f"Across the {n_images} example images ({params['encoder']}, "
        f"{params['n_samples']} samples per level)",
        fontsize=11,
        y=1.03,
    )
    return fig, data


def compute_alignment(env: dict, results: dict, areas: list, k: int) -> dict:
    """
    Per noise level and example image, the subspace alignment of the top-k eigenvectors
    of M_t(x) of each area with those of M_t^env(x), (1/k) ‖V_tᵀ V_env‖², between 0
    (orthogonal) and 1 (the same subspace): {"log_t": [J], area: [images, J]}.
    """
    # the image names, and the noise levels
    names = list(env)
    alignment = {"log_t": env[names[0]].log_tau}

    # the alignment of each area with the stimulus features, image by image, level by
    # level
    for area in areas:
        alignment[area] = torch.tensor(
            [
                [
                    subspace_alignment(
                        results[area][name].eigenvectors[level],
                        env[name].eigenvectors[level],
                        top_k=k,
                    )
                    for level in range(len(alignment["log_t"]))
                ]
                for name in names
            ]
        )
    return alignment


def plot_alignment(alignment: dict, params: dict, k: int) -> tuple[plt.Figure, dict]:
    """
    The alignment of the top-k eigenvectors of M_t(x) with those of M_t^env(x) against
    tau = log t, one panel per area: every image (thin) and their mean (thick), with
    the alignment of random subspaces, k / P, for reference. Returns the figure and the
    data it shows.
    """
    # the data drawn: per area the alignment of each image and the mean
    areas = params["areas"]
    data = {"log_t": alignment["log_t"].numpy()}
    for area in areas:
        data[area] = {
            "images": alignment[area].numpy(),
            "mean": alignment[area].mean(dim=0).numpy(),
        }

    # one panel per area, side by side, on the same scale
    fig, axes = plt.subplots(
        1, len(areas), figsize=(5.5 * len(areas), 3.6), dpi=params["screen_dpi"]
    )
    chance = k / 64**2
    for ax, area in zip(axes, areas):
        # every image (thin), and their mean (thick)
        ax.plot(data["log_t"], data[area]["images"].T, color=BLOCK_COLORS[area], lw=0.5)
        ax.plot(data["log_t"], data[area]["mean"], color=BLOCK_COLORS[area], lw=2.5)

        # random subspaces, and the labels
        ax.axhline(chance, color="0.5", ls=":", lw=1)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(r"$\tau = \log t$")
        ax.set_ylabel(f"alignment of the top {k}\neigenvectors, {area} vs stimulus")

    # the panel letters, and the title
    for letter, ax in zip("ABCD", axes):
        panel_label(fig, ax, letter)
    fig.suptitle(
        r"Alignment of $M_t(x)$ with the stimulus features $M_t^{\rm env}(x)$"
        f" ({params['encoder']}, {alignment[areas[0]].shape[0]} images; dotted: "
        "random subspaces)",
        fontsize=11,
        y=1.03,
    )
    return fig, data
