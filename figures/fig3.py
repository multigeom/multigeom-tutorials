"""
Figure 3: multi-scale geometry of the V1 and V4 population models (Inception
encoder).

  A    model schematic. Drawn by hand and NOT produced here; the left of the
       top row is left empty for it.
  B-C  measured versus predicted test responses, all 50 neurons and all test
       images pooled, for V1 (B) and V4 (C). The title gives the mean
       per-neuron Pearson correlation. Source: Stage-1 checkpoints
       (test_responses, test_predictions, per_neuron_test_corr).
  D
      the example image with the V1 (blue) and V4 (yellow) receptive-field regions
      outlined. Each neuron's region is where its Gaussian readout mask is at least half
      its peak; the outline is the boundary of the union over the 50 neurons (see
      `multigeom.encoders.receptive_field` for the mapping from readout coordinates to
      pixels).
  E    leading eigenvectors of G(x) for the example image, V1 (top) and V4
       (bottom). Source: Stage-4 per-image files.
  F    same as E for J(x).

Eigenvectors are drawn with a symmetric diverging colour map (red positive,
blue negative), each scaled to its own 99th percentile of |v|; their sign is
arbitrary.

Usage:
    python -m figures.fig3 --runs-dir runs --out-dir figures/out --img-idx 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch


from multigeom.encoders.receptive_field import readout_receptive_field

from .style import (
    AREA_COLORS,
    AREAS,
    IMAGE_SIZE,
    METRIC_TITLES,
    RF_COLORS,
    as_numpy,
    blank,
    encoder_checkpoint,
    panel_label,
    per_image,
    plt,
    save_figure,
    show_eigenvector,
    skip,
)

SCRIPT = "fig3"
ENCODER = "inception"
# Feature-map side length per backbone, used only if a checkpoint lacks the
# readout grid buffer (multigeom/encoders: out_spatial).
FEATURE_SIDE = {"inception": 12, "resnet50": 16, "onoff": 64}


def parse_args(argv: list[str] | None = None):
    """Command-line arguments of `python -m figures.fig3`."""
    p = argparse.ArgumentParser()
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="figures/out")
    p.add_argument(
        "--img-idx", type=int, default=0, help="example image (Stage-4 index)"
    )
    p.add_argument("--n-eigvecs", type=int, default=4)
    return p.parse_args(argv)


def panel_scatter(ax, ckpt: dict | None, area: str) -> None:
    """Scatter of measured against predicted test responses of one area (Fig. 3B-C)."""
    if ckpt is None or "test_predictions" not in ckpt:
        if ckpt is not None:
            skip(SCRIPT, f"{area} scatter: checkpoint has no test_predictions")
        blank(ax)
        return
    pred = as_numpy(ckpt["test_predictions"]).ravel()
    meas = as_numpy(ckpt["test_responses"]).ravel()
    r_mean = float(np.mean(as_numpy(ckpt["per_neuron_test_corr"])))
    ax.scatter(
        meas,
        pred,
        s=4,
        alpha=0.4,
        c=AREA_COLORS[area],
        edgecolors="none",
        rasterized=True,
    )
    lo = min(meas.min(), pred.min())
    hi = max(meas.max(), pred.max())
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    ax.plot(lim, lim, "--", color="black", lw=1.0)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect("equal")
    ax.set_xlabel("measured response")
    ax.set_ylabel("predicted response")
    ax.set_title(rf"{area}   $\bar r$ = {r_mean:.2f}", fontsize=12)


def panel_rf(ax, image, ckpts: dict) -> None:
    """The image with the receptive-field regions of V1 and V4 outlined (Fig. 3D)."""
    if image is None:
        blank(ax)
        return
    ax.imshow(
        as_numpy(image).reshape(IMAGE_SIZE, IMAGE_SIZE),
        cmap="gray",
        vmin=-1,
        vmax=1,
        extent=(0, IMAGE_SIZE, IMAGE_SIZE, 0),
    )
    x_text = 0.0
    for area in AREAS:
        ckpt = ckpts.get(area)
        if ckpt is None:
            continue
        field, coords = readout_receptive_field(
            ckpt["model_state_dict"], feature_side=FEATURE_SIDE[ENCODER]
        )
        ax.contour(
            coords,
            coords,
            field,
            levels=[0.5],
            colors=[RF_COLORS[area]],
            linewidths=2.0,
        )
        ax.text(
            x_text,
            -0.03,
            f"{area} RFs",
            transform=ax.transAxes,
            color=RF_COLORS[area],
            fontsize=11,
            fontweight="bold",
            va="top",
        )
        x_text += 0.5
    ax.set_xlim(0, IMAGE_SIZE)
    ax.set_ylim(IMAGE_SIZE, 0)
    ax.set_xticks([])
    ax.set_yticks([])


def panel_eigvecs(fig, spec, records: dict, metric: str, n_show: int):
    """Two rows (V1, V4) of the leading `n_show` eigenvectors of one metric."""
    inner = spec.subgridspec(
        3, n_show, height_ratios=[0.12, 1, 1], hspace=0.25, wspace=0.04
    )
    title_ax = fig.add_subplot(inner[0, :])
    title_ax.axis("off")
    title_ax.text(
        0.5,
        0.0,
        METRIC_TITLES[metric],
        ha="center",
        va="bottom",
        fontsize=13,
        transform=title_ax.transAxes,
    )
    first = None
    for r, area in enumerate(AREAS):
        rec = records.get(area)
        vecs = None if rec is None else rec.get(f"{metric}_eigenvectors")
        for k in range(n_show):
            ax = fig.add_subplot(inner[r + 1, k])
            first = first or ax
            if vecs is None or k >= len(vecs):
                blank(ax)
            else:
                show_eigenvector(ax, vecs[k])
            if k == 0:
                ax.set_ylabel(area, fontsize=11)
    return first


def main(argv: list[str] | None = None) -> list[Path]:
    """
    Draw Figure 3 from the pipeline outputs and write it (command-line entry point).
    """
    args = parse_args(argv)
    runs_dir = Path(args.runs_dir)
    ckpts = {
        area: encoder_checkpoint(runs_dir, ENCODER, area, SCRIPT) for area in AREAS
    }
    records = {
        area: per_image(runs_dir, ENCODER, area, args.img_idx, SCRIPT) for area in AREAS
    }
    image = next((rec["image"] for rec in records.values() if rec is not None), None)

    fig = plt.figure(figsize=(15, 7.6))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.05], hspace=0.28)
    top = outer[0].subgridspec(1, 4, width_ratios=[2.2, 1, 1, 0.1], wspace=0.35)
    ax_b = fig.add_subplot(top[1])
    ax_c = fig.add_subplot(top[2])
    panel_scatter(ax_b, ckpts["V1"], "V1")
    panel_scatter(ax_c, ckpts["V4"], "V4")

    bottom = outer[1].subgridspec(1, 3, width_ratios=[1.0, 2.2, 2.2], wspace=0.15)
    ax_d = fig.add_subplot(bottom[0])
    panel_rf(ax_d, image, ckpts)
    ax_e = panel_eigvecs(fig, bottom[1], records, "G", args.n_eigvecs)
    ax_f = panel_eigvecs(fig, bottom[2], records, "J", args.n_eigvecs)

    for letter, ax in zip("BCDEF", (ax_b, ax_c, ax_d, ax_e, ax_f)):
        panel_label(fig, ax, letter)
    return save_figure(fig, Path(args.out_dir), f"fig3_img{args.img_idx:04d}")


if __name__ == "__main__":
    main()
