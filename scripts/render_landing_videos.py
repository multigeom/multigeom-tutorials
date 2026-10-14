"""
Render the videos of the landing page (multigeom.github.io) from the stored fixtures,
without a GPU.

    eigenvector_walk.mp4
        the example image of paper Fig. 3D moved back and forth along the leading
        eigenvector of G(x) (top row) and of J(x) (bottom row), for the models fitted to
        V1 (left) and V4 (right)
    metric_field.mp4
        paper Fig. 5: the metric ellipses of Inception and ResNet-50 appear one grid
        point at a time, for G(x) (top) and J(x) (bottom), V1 (left) and V4 (right)

Needs numpy, matplotlib and imageio-ffmpeg (it ships an ffmpeg binary).

    python scripts/render_landing_videos.py --out-dir
    ../multigeom.github.io/docs/assets/videos
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "fig3"
AREAS = ("V1", "V4")
METRICS = {"G": "multi-scale Fisher G(x)", "J": "Fisher information J(x)"}


def eigenvector_walk(
    out_dir: Path, amplitude: float = 10.0, n_frames: int = 72, fps: int = 24
) -> Path:
    """
    `amplitude`: size of the step along a unit-norm eigenvector (about 0.16 per pixel at
    10).
    """
    import imageio_ffmpeg

    data = np.load(FIXTURES / "fig3_def_data.npz")
    x0 = data["image"].reshape(64, 64)
    fig, axes = plt.subplots(2, 2, figsize=(6.4, 6.8), dpi=100)
    images = {}
    for r, metric in enumerate(METRICS):
        for c, area in enumerate(AREAS):
            ax = axes[r, c]
            images[(metric, area)] = ax.imshow(x0, cmap="gray", vmin=-1, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(f"{metric}(x), {area}", fontsize=12)
    label = fig.text(0.5, 0.015, "", ha="center", fontsize=12)
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.94, bottom=0.07, wspace=0.04, hspace=0.12
    )

    out = out_dir / "eigenvector_walk.mp4"
    out_dir.mkdir(parents=True, exist_ok=True)
    w, h = fig.canvas.get_width_height()
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "26",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in range(n_frames):
        t = amplitude * np.sin(2 * np.pi * f / n_frames)
        for (metric, area), im in images.items():
            v = data[f"{area}_{metric}"][0].reshape(64, 64)
            im.set_data(np.clip(x0 + t * v, -1, 1))
        label.set_text(f"image + t × leading eigenvector,   t = {t:+.1f}")
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    plt.close(fig)
    return out


def _encode(fig, out: Path, frames, fps: int) -> Path:
    """
    Pipe the frames drawn by `frames()` (a generator that updates `fig` and yields) to
    an H.264 mp4.
    """
    import imageio_ffmpeg

    out.parent.mkdir(parents=True, exist_ok=True)
    w, h = fig.canvas.get_width_height()
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgba",
        "-s",
        f"{w}x{h}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-an",
        "-vf",
        "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-crf",
        "26",
        "-movflags",
        "+faststart",
        str(out),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for _ in frames():
        fig.canvas.draw()
        proc.stdin.write(np.asarray(fig.canvas.buffer_rgba()).tobytes())
    proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    plt.close(fig)
    return out


def metric_field(out_dir: Path, fps: int = 6, hold_frames: int = 12) -> Path:
    """
    The ellipses of paper Fig. 5 appear in raster order over the perturbation grid (the
    stored 5 x 5 fixture).
    """
    import torch
    from matplotlib.lines import Line2D

    from multigeom.viz.fig5 import draw_field
    from multigeom.viz.style import ENCODER_COLORS, ENCODER_LABELS_LONG

    data = np.load(
        Path(__file__).resolve().parents[1]
        / "tests"
        / "fixtures"
        / "fig5"
        / "metric_field.npz"
    )
    encoders = ("inception", "resnet50")
    coeffs = data["grid_coefficients"]
    n = len(coeffs) ** 2
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.6), dpi=100)
    patches = {}
    for r, metric in enumerate(METRICS):
        for c, area in enumerate(AREAS):
            ax = axes[r, c]
            rec = {
                "grid_coefficients": coeffs,
                "encoders": list(encoders),
                "matrices_2x2": {
                    e: torch.from_numpy(data[f"{metric}_{area}_{e}"]) for e in encoders
                },
            }
            draw_field(ax, rec, 0.9)
            ax.set_title(f"{metric}(x), {area}", fontsize=12)
            patches[(metric, area)] = list(ax.patches)[: 2 * n]
    handles = [
        Line2D(
            [0],
            [0],
            color=ENCODER_COLORS[e],
            lw=1.8,
            ls=ls,
            label=ENCODER_LABELS_LONG[e],
        )
        for e, ls in zip(encoders, ("-", "--"))
    ]
    fig.legend(
        handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=12
    )  # below the axes
    label = fig.text(0.5, 0.075, "", ha="center", fontsize=11)
    fig.subplots_adjust(
        left=0.02, right=0.98, top=0.95, bottom=0.14, wspace=0.06, hspace=0.16
    )

    def frames():
        for k in range(n + 1 + hold_frames):
            shown = min(k, n)
            for ps in patches.values():
                for i, patch in enumerate(ps):
                    patch.set_visible(
                        i % n < shown
                    )  # patches: all points of the first model, then the second
            label.set_text(f"grid of perturbed images:  {shown}/{n} points")
            yield

    return _encode(fig, out_dir / "metric_field.mp4", frames, fps)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--out-dir", default="videos")
    args = p.parse_args(argv)
    print(eigenvector_walk(Path(args.out_dir)))
    print(metric_field(Path(args.out_dir)))


if __name__ == "__main__":
    main()
