"""
Check that multigeom, on Baptiste's retina models (open-retina), reproduces his figures
of the Fisher information J(x) and the multi-scale Fisher information G(x) of the
mosaics of W3 and tOFFalpha cells (tests/fixtures/retina/, described in its README).

Integration tests of `multigeom.encoders.retina` and of the Poisson noise of the
responses (`multigeom.geometry.noise`): they load the mosaic of a cell, its diffusion
model and the test images of open-retina's run_lorenzi, compute J(x) or G(x), and
compare the result with the tiles of the reference figure, turned back into values
through its colour scale. Each run also redraws the figure (`multigeom.plotting.retina`)
next to the reference, in tests/fixtures/retina/ (<name>.png next to
<name>_reference.png), to compare them by eye. Which cells and images each figure shows
is in tests/fixtures/retina/README.md.

They need the `openretina` package of the private repository
steevelaquitaine/open-retina-multigeom (open-retina with Baptiste's information-geometry
code), the retina weights and the TinyImageNet images (see
tests/fixtures/retina/README.md): they are skipped without them, and on GitHub.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from retina_helpers import (  # noqa: E402
    FRAMES,
    NORMALIZER,
    FIXTURES,
    RUNS,
    find_tiles,
    get_correlation,
    get_peak,
    get_tile,
    get_values,
    have_retina,
    reduce,
)

# not the tests of a notebook: listed after them in tests/README.md
pytestmark = [
    pytest.mark.slow,
    pytest.mark.gpu,
    pytest.mark.skipif(
        not have_retina(),
        reason="needs openretina, the retina weights and TinyImageNet",
    ),
]

# the areas (runs/<stage>/lorenzi/<area>/) of the cells of the figures
AREAS = {"W3": "retina-W3_176-exp7", "tOFFalpha": "retina-tOFFalpha_17-exp5"}

# test_retina_j_x.png and test_retina_g_x.png: J(x) and G(x) of W3 cell 176-exp7 on the
# 6 test images; the shares of e1-e5 printed on their tiles, in %
SHARES_J = [
    [9, 7, 5, 4, 3],
    [6, 6, 5, 5, 4],
    [4, 3, 3, 3, 3],
    [7, 7, 7, 3, 3],
    [4, 4, 4, 3, 3],
    [5, 5, 4, 4, 4],
]
SHARES_G = [
    [51, 16, 7, 5, 3],
    [54, 18, 5, 2, 2],
    [53, 15, 6, 3, 2],
    [55, 12, 6, 4, 3],
    [52, 17, 6, 3, 2],
    [60, 13, 6, 2, 2],
]

# test_retina_jx_gx.png (panel A of Baptiste's test_retina1.png): test image 2 of 6
# (the bus), and its rows: G(x) of W3 cell
# 176-exp7, G(x) of tOFFalpha cell 17-exp5, J(x) of tOFFalpha cell 17-exp5, with their
# shares; its frames are blue (G) and orange (J)
INDEX_A = 2
ROWS_A = [("W3", "G"), ("tOFFalpha", "G"), ("tOFFalpha", "J")]
SHARES_A = [[53, 15, 6, 3, 2], [37, 9, 7, 6, 5], [2, 1, 1, 1, 1]]
FRAME_COLOURS_A = ((85 / 255, 129 / 255, 211 / 255), (213 / 255, 145 / 255, 81 / 255))

# Tolerances. The figure's pixels fall up to a pixel off the image's, and rendering
# blurs the small receptive fields: our own eigenvectors, rendered the same way, keep a
# correlation of about 0.98 with themselves.
#  - J(x) is deterministic. Eigenvalues a few % apart can swap order, so each reference
#    eigenvector is matched among our 10 leading ones; a receptive field in a tile's
#    corner, partly under the printed share, lowers single tiles to 0.67.
#  - G(x) comes from random draws (two of our runs differ as much as two of
#    open-retina's): its two leading eigenvectors are compared rank by rank, with their
#    shares of the 20 leading eigenvalues; the next ones carry a few % each and swap
#    order from run to run. Over 3 seeds on the bus, e1 correlates at 0.94-0.98 and e2
#    at 0.92-0.95, and the share of e1 varies by up to 2 points.
STIMULUS_MIN_CORRELATION = 0.95
N_CANDIDATES = 10
J_MIN_CORRELATION = 0.6
J_MEDIAN_CORRELATION = 0.95
J_SHARE_TOLERANCE = 1
J_PEAK_DISTANCE = 3
G_MIN_CORRELATION = (0.9, 0.8)
G_SHARE_TOLERANCE = (5, 3)


@pytest.fixture(scope="module")
def device() -> torch.device:
    """The GPU."""
    return torch.device("cuda")


def load_cell(cell_type: str, device: torch.device):
    """The models of the cell of `cell_type`, and the 6 test images."""
    from multigeom.encoders.retina import load_retina, load_retina_test_images

    # the area's models, and the images in their units
    models = load_retina(
        AREAS[cell_type], RUNS, FRAMES, device, normalizer_cache=NORMALIZER
    )
    return models, load_retina_test_images(models, FRAMES, n=6)


def compute_metric(models, image: torch.Tensor, metric: str, seed: int = 0):
    """
    J(x) of one image [1, 120, 120], all its eigenpairs, or G(x) with the settings of
    open-retina's run_lorenzi (150 samples per noise level, 20 eigenpairs).
    """
    from multigeom.config import load_config
    from multigeom.geometry import compute_g
    from multigeom.geometry.fisher import fisher_eigendecomposition

    # J(x): the Poisson Fisher information of the mosaic
    device = models.alphas_cumprod.device
    if metric == "J":
        return fisher_eigendecomposition(
            image[None].to(device),
            models.encoder,
            sigma=1.0,
            top_k=models.encoder.n_neurons,
            device=device,
            validate=False,
            noise=models.noise,
        )

    # G(x): with the Poisson noise of the responses
    elif metric == "G":
        cfg = copy.deepcopy(load_config("estimator"))
        cfg.monte_carlo.n_samples, cfg.top_k = 150, 20
        cfg.monte_carlo.noise = models.noise
        return compute_g(
            image[None].to(device),
            models.encoder,
            models.unet,
            models.alphas_cumprod,
            cfg,
            device,
            generator=torch.Generator(device).manual_seed(seed),
        )
    else:
        raise ValueError(f"unknown metric {metric!r}: G or J")


def check_row(picture, span_y, columns, output, metric: str, shares: list) -> list:
    """
    Compare one row of eigenvector tiles of a reference figure with our G(x) or J(x):
    G rank by rank for e1 and e2, with their shares; J tile by tile against our 10
    leading eigenvectors, with the shares of e1-e5. Returns the correlations.
    """
    vectors = output.eigenvectors[:N_CANDIDATES].reshape(-1, 120, 120).cpu().numpy()
    references = [
        reduce(get_values(get_tile(picture, span_y, span_x)), 120) for span_x in columns
    ]
    values = output.eigenvalues.double().cpu()
    ours = (100 * values / values.sum())[:5].numpy()

    # G(x): e1 and e2, rank by rank, and their shares
    if metric == "G":
        correlations = [get_correlation(references[k], vectors[k]) for k in range(2)]
        for k in range(2):
            assert correlations[k] >= G_MIN_CORRELATION[k], (k + 1, correlations)
            assert abs(ours[k] - shares[k]) <= G_SHARE_TOLERANCE[k] + 0.5, (k + 1, ours)
        return correlations

    # J(x): every tile against the best of our leading eigenvectors, and the shares
    elif metric == "J":
        correlations = [max(get_correlation(r, v) for v in vectors) for r in references]
        assert min(correlations) >= J_MIN_CORRELATION, correlations
        np.testing.assert_allclose(ours, shares, atol=J_SHARE_TOLERANCE + 0.5)
        return correlations
    else:
        raise ValueError(f"unknown metric {metric!r}: G or J")


def check_peaks(picture, span_y, columns, output, shares: list) -> None:
    """
    Compare the receptive fields of a row of J(x) tiles with ours: the peak of each
    tile (its largest |value| away from the 3-pixel border) within 3 pixels of the peak
    of one of our 10 leading eigenvectors, and the shares of e1-e5.
    """
    vectors = output.eigenvectors[:N_CANDIDATES].reshape(-1, 120, 120).cpu().numpy()
    peaks = [get_peak(vector) for vector in vectors]
    for k, span_x in enumerate(columns):
        peak = get_peak(reduce(get_values(get_tile(picture, span_y, span_x)), 120))
        distance = min(np.hypot(*np.subtract(peak, ours)) for ours in peaks)
        assert distance <= J_PEAK_DISTANCE, (k + 2, peak, distance)

    # the shares of e1-e5 in the sum of the eigenvalues, in %
    values = output.eigenvalues.double().cpu()
    ours = (100 * values / values.sum())[:5].numpy()
    np.testing.assert_allclose(ours, shares, atol=J_SHARE_TOLERANCE + 0.5)


def check_stimulus(picture, span_y, span_x, image: torch.Tensor) -> None:
    """The image's tile against our image, whatever the grey scale."""
    stimulus = reduce(get_tile(picture, span_y, span_x).mean(axis=2), 120)
    correlation = get_correlation(stimulus, image[0].cpu().numpy())
    assert correlation >= STIMULUS_MIN_CORRELATION, correlation


@pytest.mark.parametrize(
    "figure, metric, shares",
    [("test_retina_j_x", "J", SHARES_J), ("test_retina_g_x", "G", SHARES_G)],
)
def test_cell_figure_is_reproduced(figure, metric, shares, device):
    """
    Verify that J(x) (test_retina_j_x.png) or G(x) (test_retina_g_x.png) of the mosaic
    of W3 cell 176-exp7, on the 6 test images of open-retina's run_lorenzi, reproduces
    the figure: each image tile (correlation ≥ 0.95); for J, each of the 5 leading
    eigenvector tiles matched among our 10 leading eigenvectors (|correlation| ≥ 0.6,
    median ≥ 0.95) and the shares of e1-e5 (within 1 percentage point); for G, the two
    leading eigenvectors rank by rank (|correlation| ≥ 0.9 and 0.8) and their shares of
    the 20 leading eigenvalues (within 5 and 3 points).
    """
    from multigeom.plotting.retina import plot_cell_summary

    # the cell, the test images, and the metric of each
    models, images = load_cell("W3", device)
    outputs = [
        compute_metric(models, image, metric, seed) for seed, image in enumerate(images)
    ]

    # our figure, redrawn in the reference's layout, next to it
    fig, _ = plot_cell_summary(images, outputs, f"ours: {models.cell}, {metric}(x)")
    fig.savefig(FIXTURES / f"{figure}.png", dpi=130)

    # the reference's tiles: 6 rows, the image then e1-e5
    picture = mpimg.imread(FIXTURES / f"{figure}_reference.png")[..., :3]
    rows, columns = find_tiles(picture)
    assert (len(rows), len(columns)) == (6, 6)

    # each row: the image, then the eigenvectors and their shares
    correlations = []
    for row, span_y in enumerate(rows):
        check_stimulus(picture, span_y, columns[0], images[row])
        correlations += check_row(
            picture, span_y, columns[1:], outputs[row], metric, shares[row]
        )

    # J(x): most eigenvectors match to the rendering's precision
    if metric == "J":
        assert np.median(correlations) >= J_MEDIAN_CORRELATION, correlations


def test_jx_gx_figure_is_reproduced(device):
    """
    Verify that test_retina_jx_gx.png (the bus, test image 2 of 6) is reproduced:
    the two leading eigenvectors of G(x) of W3 cell 176-exp7 and of tOFFalpha cell
    17-exp5, rank by rank (|correlation| ≥ 0.9 and 0.8; shares of the 20 leading
    eigenvalues within 5 and 3 percentage points), and the receptive fields of J(x)
    of tOFFalpha cell 17-exp5: the peak of e2-e5 within 3 pixels of a peak of our 10
    leading eigenvectors (shares of e1-e5 within 1 point). The figure is small (its
    tiles about 116 pixels for the 120 of the image), which blurs the few-pixel
    receptive fields of J(x) too much to compare them pixel by pixel; e1's is under
    the printed share.
    """
    from multigeom.plotting.retina import plot_cell_summary

    # the reference's tiles: 3 rows, 5 columns (the frames are 1 pixel, antialiased)
    picture = mpimg.imread(FIXTURES / "test_retina_jx_gx_reference.png")[..., :3]
    rows, columns = find_tiles(picture, FRAME_COLOURS_A, fraction=0.25, tolerance=0.25)
    assert (len(rows), len(columns)) == (3, 5)

    # each row: its cell's models, the bus, the metric, and the comparison
    images, outputs = [], []
    for row, (cell_type, metric) in enumerate(ROWS_A):
        models, test_images = load_cell(cell_type, device)
        output = compute_metric(models, test_images[INDEX_A], metric)
        if metric == "G":
            check_row(picture, rows[row], columns, output, metric, SHARES_A[row])
        else:
            check_peaks(picture, rows[row], columns[1:], output, SHARES_A[row])
        images.append(test_images[INDEX_A])
        outputs.append(output)

    # our panel, redrawn in the reference's layout, next to it
    labels = [f"{cell_type}\n{metric}(x)" for cell_type, metric in ROWS_A]
    fig, _ = plot_cell_summary(torch.stack(images), outputs, "ours: the bus", labels)
    fig.savefig(FIXTURES / "test_retina_jx_gx.png", dpi=130)
