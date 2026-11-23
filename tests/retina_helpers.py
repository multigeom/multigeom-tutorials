"""
Helpers of the retina tests (tests/test_retina.py): where the retina models' checkpoints
are (runs/, as those of V1 and V4; the code is the `openretina` package of the private
repository steevelaquitaine/open-retina-multigeom, see tests/fixtures/retina/README.md),
and the reading of Baptiste's reference figures (tests/fixtures/retina/): their tiles,
cut out at the axes frames, turned back into values through the colour scale, and
compared with eigenvectors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
# Baptiste's figures (copies of assets/test_retina/*.png, as <name>_reference.png) and
# ours, redrawn by each run of the tests (<name>.png), side by side to compare
FIXTURES = REPO / "tests" / "fixtures" / "retina"

# the checkpoints (runs/, downloaded from the Hugging Face repository
# steevelaquitaine/multigeom), the areas of the retina models the tests use, the
# TinyImageNet training images, and the cached image units
RUNS = REPO / "runs"
AREAS = ("retina-W3_176-exp7", "retina-tOFFalpha_17-exp5")
FRAMES = REPO / "data" / "tiny-imagenet-200" / "train"
NORMALIZER = REPO / "data" / "02_intermediate" / "retina" / "normalizer.pt"


def have_retina() -> bool:
    """Whether openretina, the retina checkpoints and TinyImageNet are here."""
    has_package = importlib.util.find_spec("openretina") is not None
    checkpoints = [
        RUNS / stage / "lorenzi" / area / name
        for area in AREAS
        for stage, name in (
            ("encoders", "best_model.pt"),
            ("diffusion", "final_model.pt"),
        )
    ]
    return has_package and FRAMES.exists() and all(c.exists() for c in checkpoints)


def find_lines(mask: np.ndarray, axis: int, fraction: float) -> list:
    """
    The frame lines of `mask` [H, W] (True on frame pixels): the rows (axis 1) or
    columns (axis 0) where more than `fraction` of the pixels are on a frame, those up
    to 4 pixels apart merged (a thick frame's two antialiased edges), as
    [(first, last)].
    """
    groups = []
    for index in np.where(mask.sum(axis=axis) > fraction * mask.shape[axis])[0]:
        if groups and index - groups[-1][1] <= 4:
            groups[-1][1] = index
        else:
            groups.append([index, index])
    return groups


def find_tiles(
    picture: np.ndarray,
    colours: tuple = (),
    fraction: float = 0.5,
    tolerance: float = 0.12,
) -> tuple[list, list]:
    """
    The tiles of a figure [H, W, 3] whose tiles are squares of one size: the rows and
    the columns of pixels inside the axes frames, as [(first, last)] pairs. The frames
    are the lines, dark or of one of `colours` (RGB in [0, 1], within `tolerance`
    summed over the channels), that cross more than `fraction` of the figure. Dark
    images can add lines, so a tile lies between two lines as far apart as the tiles
    are high.
    """
    # the frame pixels: dark, or of a frame colour
    mask = picture.mean(axis=2) < 0.3
    for colour in colours:
        mask |= np.abs(picture - np.asarray(colour)).sum(axis=2) < tolerance
    rows = find_lines(mask, 1, fraction)
    columns = find_lines(mask, 0, fraction)

    # the side of the tiles: the most common gap between consecutive lines (over 100
    # pixels, within 3 pixels)
    gaps = np.array(
        [
            b[0] - a[1] - 1
            for lines in (rows, columns)
            for a, b in zip(lines[:-1], lines[1:])
            if b[0] - a[1] - 1 > 100
        ]
    )
    side = max(gaps, key=lambda gap: (np.abs(gaps - gap) <= 3).sum())

    # from first to last, each line with a line one tile further
    def get_spans(lines: list) -> list:
        """The spans between two lines one tile apart, not overlapping."""
        spans, start = [], -1
        for a in lines:
            ends = [b for b in lines if abs(b[0] - a[1] - 1 - side) <= 3]
            if a[1] > start and ends:
                spans.append((a[1] + 1, ends[0][0] - 1))
                start = ends[0][0] - 1
        return spans

    return get_spans(rows), get_spans(columns)


def get_tile(picture: np.ndarray, span_y: tuple, span_x: tuple) -> np.ndarray:
    """The pixels of one tile [h, w, 3]."""
    return picture[span_y[0] : span_y[1] + 1, span_x[0] : span_x[1] + 1]


def get_values(tile: np.ndarray, cmap: str = "RdBu_r") -> np.ndarray:
    """
    The values in [-1, 1] that the colours of an eigenvector tile stand for, through the
    colour scale `cmap`; NaN where the colour is not on the scale (text).
    """
    import matplotlib

    # the nearest colour of the scale, if near enough
    scale = matplotlib.colormaps[cmap](np.linspace(0, 1, 256))[:, :3]
    distance = ((tile[..., None, :] - scale[None, None]) ** 2).sum(axis=-1)
    values = np.linspace(-1, 1, 256)[distance.argmin(axis=-1)]
    values[distance.min(axis=-1) > 0.01] = np.nan
    return values


def reduce(values: np.ndarray, side: int) -> np.ndarray:
    """The mean of a tile's values over each of the side x side image pixels."""
    height, width = values.shape
    rows = np.arange(height) * side // height
    columns = np.arange(width) * side // width
    valid = ~np.isnan(values)
    total, count = np.zeros((side, side)), np.zeros((side, side))
    np.add.at(total, (rows[:, None], columns[None, :]), np.where(valid, values, 0))
    np.add.at(count, (rows[:, None], columns[None, :]), valid)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def get_correlation(reference: np.ndarray, image: np.ndarray, shift: int = 1) -> float:
    """
    The largest |correlation| between a reference tile's values and an image of the
    same size shifted by up to `shift` pixels (the figure's pixels fall up to a pixel
    off the image's), over the reference's valid pixels.
    """
    valid = ~np.isnan(reference)
    best = 0.0
    for dy in range(-shift, shift + 1):
        for dx in range(-shift, shift + 1):
            moved = np.roll(image, (dy, dx), axis=(0, 1))
            best = max(best, abs(np.corrcoef(reference[valid], moved[valid])[0, 1]))
    return best


def get_peak(values: np.ndarray, border: int = 3) -> tuple:
    """
    Where the largest |value| of an image is, away from its `border` pixels (where a
    figure's frame blends into the tile), as (row, column); NaN values are left out.
    """
    inner = np.abs(values).astype(float)
    inner[:border], inner[-border:] = np.nan, np.nan
    inner[:, :border], inner[:, -border:] = np.nan, np.nan
    return np.unravel_index(np.nanargmax(inner), inner.shape)
