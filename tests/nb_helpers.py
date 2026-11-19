"""
Helpers to run cells of the tutorial notebooks inside the tests (notebook 3 by default,
notebook 5 for Fig. 5, notebook 6 for Fig. 6).
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = REPO / "notebooks" / "3_leading_eigenvectors_of_G.ipynb"
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fig3"

# A unique line of each notebook code cell used by the tests. The plot cells rely on the
# notebook's setup cell for their imports (plt, panel_rf, ...), supplied by
# `render_plot_cell`.
SETUP = "# import built-in packages"
FUNCTIONS = "def load_models("
LOAD_MODELS = "alphas_cumprod = scheduler.alphas_cumprod"
COMPUTE = "ckpts, records = {}, {}"
PLOT_FIG3 = "panel_rf("
PLOT_FIG4 = "panel_eigvecs_fig4("

AREAS = ("V1", "V4")
ENCODERS = ("inception", "resnet50")
N_FIG3 = 4  # eigenvectors of the Fig. 3 encoder stored per metric and area
N_FIG4_REF = 4  # ... of the reference encoder (inception) for Fig. 4B-C
N_FIG4_POOL = 30  # ... of the other encoders: the notebook searches their first 30


NOTEBOOK3 = REPO / "notebooks" / "5_local_metric_field.ipynb"
# Lines of the code cells of notebook 5 (parameters, launch, wait, plot).
SETUP3 = "from multigeom.pipeline import metric_field_job as lmf"
LAUNCH3 = "lmf.launch(job)"
WAIT3 = "lmf.wait(job"
PLOT3 = "show_figure(make_figure(lmf.load_records(job)))"


NOTEBOOK4 = REPO / "notebooks" / "6_bures_wasserstein.ipynb"
# Lines of the code cells of notebook 6 (setup, Stage 5, ordering check, plot).
SETUP4 = "from multigeom.pipeline import alignment as bw"
COMPUTE4 = "bw.compute_metrics(RUNS_DIR"
ORDER4 = "bw.print_ordering(metrics)"
PLOT4 = "show_figure(plot_bures_wasserstein(metrics))"


def notebook_cell(marker: str, notebook: Path | None = None) -> str:
    """Source of the code cell containing `marker`, without IPython magics (%%time)."""
    notebook = notebook or NOTEBOOK
    nb = json.loads(notebook.read_text())
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if marker in src:
            return "\n".join(
                l for l in src.splitlines() if not l.lstrip().startswith("%%")
            )
    raise KeyError(f"no code cell containing {marker!r} in {notebook}")


def run_cells(
    namespace: dict,
    *markers: str,
    notebook: Path | None = None,
    overrides: dict | None = None,
) -> dict:
    """
    Execute the notebook cells containing `markers`, in order, in `namespace`.
    `overrides` replaces the value of top-level assignments `NAME = value` in the cells
    (for example {"GRID_SIZE": "3"}), to run a smaller job.
    """
    for marker in markers:
        source = notebook_cell(marker, notebook)
        # The notebooks reload the modules they use ("importlib.reload(...)") to pick up
        # edits; in a test session that would redefine classes under the feet of the
        # other tests, so those lines are dropped.
        source = "\n".join(
            l
            for l in source.splitlines()
            if not l.lstrip().startswith("importlib.reload(")
        )
        for name, value in (overrides or {}).items():
            source = override_assignment(source, name, value)
        exec(compile(source, f"<notebook cell: {marker}>", "exec"), namespace)
    return namespace


def override_assignment(source: str, name: str, value: str) -> str:
    """
    Replace the right-hand side of the top-level line `name = ...` (comment kept) by
    `value`; unchanged if absent.
    """
    import re

    return re.sub(
        rf"^({re.escape(name)}\s*=\s*)(.*?)(\s*#.*)?$",
        lambda m: f"{m.group(1)}{value}{m.group(3) or ''}",
        source,
        flags=re.M,
    )


def fixture_inputs() -> tuple[dict, dict]:
    """
    (ckpts, records), keyed by (encoder, area) like the notebook's compute cell, from
    the stored fixtures. ckpts hold the readout parameters of the Fig. 3 encoder
    (inception); records hold the image and the leading eigenvectors of G and J.
    """
    import numpy as np
    import torch

    d3 = np.load(FIXTURES / "fig3_def_data.npz")
    d4 = np.load(FIXTURES / "fig4_bc_data.npz")
    image = torch.from_numpy(d3["image"]).reshape(1, 1, 64, 64)
    ckpts, records = {}, {}
    for area in AREAS:
        ckpts[("inception", area)] = {
            "model_state_dict": {
                "readout.mu": torch.from_numpy(d3[f"{area}_mu"]),
                "readout.sigma": torch.from_numpy(d3[f"{area}_sigma"]),
                "readout.grid_x": torch.from_numpy(d3[f"{area}_grid_x"]),
            }
        }
        for enc in ENCODERS:
            records[(enc, area)] = {
                "image": image,
                "G_eigenvectors": torch.from_numpy(d4[f"{enc}_{area}_G"]),
                "J_eigenvectors": torch.from_numpy(d4[f"{enc}_{area}_J"]),
            }
    # Fig. 3 uses the first N_FIG3 eigenvectors; they are the same numbers as in
    # fig3_def_data.npz
    for area in AREAS:
        for metric in ("G", "J"):
            np.testing.assert_array_equal(
                records[("inception", area)][f"{metric}_eigenvectors"][:N_FIG3].numpy(),
                d3[f"{area}_{metric}"],
            )
    return ckpts, records


def render_plot_cell(
    ckpts: dict, records: dict, out_png: Path, which: str = "fig3"
) -> None:
    """
    Run the notebook's plot cell for Fig. 3D-F ("fig3") or Fig. 4B-C ("fig4") and save
    the figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figures.style import panel_label
    from figures.fig3 import panel_eigvecs, panel_rf
    from figures.fig4 import panel_eigvecs as panel_eigvecs_fig4

    plt.close("all")
    ns = run_cells(
        {
            "plt": plt,
            "ckpts": ckpts,
            "records": records,
            "ENC": "inception",
            "AREAS": list(AREAS),
            "panel_rf": panel_rf,
            "panel_eigvecs": panel_eigvecs,
            "panel_eigvecs_fig4": panel_eigvecs_fig4,
            "panel_label": panel_label,
        },
        {"fig3": PLOT_FIG3, "fig4": PLOT_FIG4}[which],
    )
    ns["fig"].savefig(out_png, dpi=100)
    plt.close("all")


# ── comparison of rendered plots ────────────────────────────────────────────
REFERENCE_PNG = {
    "fig3": FIXTURES / "fig3_def_reference.png",
    "fig4": FIXTURES / "fig4_bc_reference.png",
}
MEAN_ABS_TOL = 1e-3  # mean |difference| over all pixels, in [0, 1]; text may differ
BAD_PIXEL_FRACTION = 2e-3  # at most this fraction of pixels may differ by more than 0.1

# Recompute checks: (metric, number of leading eigenvectors compared, minimum |cosine|
# with the stored one). Measured, CPU against the GPU fixture (inception): J |cos| >=
# 0.9958 for all four eigenvectors; G |cos| = 0.96 (V1) and 0.98 (V4) for the leading
# one, but down to 0.30 for the 3rd (V4).
CHECKS = (("J", 4, 0.99), ("G", 1, 0.9))


def image_difference(a, b) -> tuple[float, float]:
    """
    (mean |a - b|, fraction of pixels with |a - b| > 0.1) of two images with values in
    [0, 1].
    """
    import numpy as np

    assert a.shape == b.shape, f"image shapes differ: {a.shape} vs {b.shape}"
    d = np.abs(a.astype(float) - b.astype(float))
    return float(d.mean()), float((d > 0.1).mean())


def plot_difference(
    which: str, ckpts: dict, records: dict, tmp_path: Path
) -> tuple[float, float]:
    """
    Render the notebook plot `which` from (ckpts, records) and compare it with its
    reference picture.
    """
    import matplotlib.image as mpimg

    out = tmp_path / f"{which}.png"
    render_plot_cell(ckpts, records, out, which)
    return image_difference(mpimg.imread(out), mpimg.imread(REFERENCE_PNG[which]))


# ── Fig. 5 fixture ──────────────────────────────────────────────────────────
FIG5_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fig5"
FIG5_ENCODERS = ("inception", "resnet50")


def fig5_records(path: Path | None = None) -> dict:
    """
    Stage-6-like records keyed by (metric, area), with the keys the Fig. 5 drawing
    needs, from the stored npz.
    """
    import numpy as np
    import torch

    data = np.load(path or FIG5_FIXTURES / "metric_field.npz")
    records = {}
    for metric in ("G", "J"):
        for area in ("V1", "V4"):
            records[(metric, area)] = {
                "grid_coefficients": torch.from_numpy(data["grid_coefficients"]),
                "encoders": list(FIG5_ENCODERS),
                "matrices_2x2": {
                    enc: torch.from_numpy(data[f"{metric}_{area}_{enc}"])
                    for enc in FIG5_ENCODERS
                },
                "x0": torch.from_numpy(data["x0"]).reshape(1, 1, 64, 64),
            }
    return records


def run_all_code_cells(
    namespace: dict,
    notebook: Path,
    overrides: dict | None = None,
    replacements: dict | None = None,
) -> dict:
    """
    Run every code cell of `notebook`, in order, in `namespace` (with `overrides` as in
    `run_cells`). `replacements` maps text to replace by other text in every cell (for
    settings that are not a plain `NAME = value` line, such as tuple assignments).
    IPython magics are dropped: `%%time` lines, and `%pip install` lines (which are only
    reached on Google Colab).
    """
    nb = json.loads(notebook.read_text())
    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        lines = []
        for line in "".join(cell["source"]).splitlines():
            if line.lstrip().startswith("%%"):
                continue
            if line.lstrip().startswith("%pip"):
                line = line[: len(line) - len(line.lstrip())] + "pass"
            lines.append(line)
        source = "\n".join(lines)
        for old, new in (replacements or {}).items():
            source = source.replace(old, new)
        for name, value in (overrides or {}).items():
            source = override_assignment(source, name, value)
        exec(compile(source, f"<{notebook.name} cell {i}>", "exec"), namespace)
    return namespace


# ── inputs of the slow tests of notebooks 7 to 10 ──────────────────────────
# the frog, the example image of notebooks 7 and 8
FROG_IMAGE = "data/tiny-imagenet-200/train/n01644900/images/n01644900_0.JPEG"


def have_inputs(data: list[str], areas: tuple = AREAS) -> bool:
    """
    Whether the data paths `data` (relative to the repository root) and the inception
    checkpoints (encoder and diffusion model) of `areas` are present: the slow tests
    recompute from them, and they are not available on GitHub.
    """
    # the data, then the two checkpoints of each area
    needed = [REPO / path for path in data]
    for area in areas:
        needed += [
            REPO / f"runs/encoders/inception/{area}/best_model.pt",
            REPO / f"runs/diffusion/inception/{area}/final_model.pt",
        ]
    return all(path.exists() for path in needed)


# ── Notebook 7: eigenvectors of G(x) at each noise level ───────────────────
NOTEBOOK7 = REPO / "notebooks" / "7_eigenvectors_of_G_per_scale.ipynb"
PER_SCALE_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "per_scale"

# Lines of the code cells of notebook 7 (setup, compute, plot); its functions are in
# figures/per_scale.py.
SETUP7 = "PARAMS = {"
COMPUTE7 = "results[area] = compute_per_scale("
PLOT7 = "fig, plotted[name] = plot_per_scale("


def per_scale_inputs(path: Path | None = None) -> tuple[dict, dict]:
    """
    (examples, results) like those of notebook 7's compute cell, for the frog only,
    from the stored fixture: examples maps "frog" to (image, seed), and results maps
    each area to {"frog": record}, with the attributes of a PerScaleOutput.
    """
    from types import SimpleNamespace

    import numpy as np
    import torch

    # the stored image, noise levels, eigenvalues and eigenvectors of the frog
    data = np.load(path or PER_SCALE_FIXTURES / "per_scale_data.npz")
    image = torch.from_numpy(data["image"]).reshape(1, 1, 64, 64)
    examples = {"frog": (image, int(data["seed"]))}

    # one record per area, eigenvectors back to float32 (stored as float16)
    results = {}
    for area in AREAS:
        record = SimpleNamespace(
            log_tau=torch.from_numpy(data["log_t"]),
            eigenvalues=torch.from_numpy(data[f"{area}_eigenvalues"]),
            eigenvectors=torch.from_numpy(
                data[f"{area}_eigenvectors"].astype(np.float32)
            ),
        )
        results[area] = {"frog": record}
    return examples, results


def render_per_scale_plot(
    examples: dict, results: dict, out_png: Path, figure_dir: Path
) -> dict:
    """
    Run notebook 7's plot cell on (examples, results), with the figures saved under
    `figure_dir`, and save the last figure drawn to `out_png` (100 dpi). Returns the
    namespace of the cell.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figures.per_scale import plot_per_scale

    # the names of the setup cell that the plot cell uses, and the settings of the plot
    plt.close("all")
    namespace = {"plt": plt, "Path": Path, "plot_per_scale": plot_per_scale}
    namespace.update(examples=examples, results=results)
    namespace["PARAMS"] = {
        "encoder": "inception",
        "areas": list(AREAS),
        "n_samples": 200,
        "figure_dir": str(figure_dir),
        "file_dpi": 100,
        "screen_dpi": 100,
        "show": ["frog"],
    }

    # the plot cell, then the figure of the last image
    run_cells(namespace, PLOT7, notebook=NOTEBOOK7)
    namespace["fig"].savefig(out_png, dpi=100)
    plt.close("all")
    return namespace


def run_notebook7_frog() -> dict:
    """
    Namespace after running notebook 7's setup and compute cells for the frog only
    (needs runs/, TinyImageNet and the multigeom with compute_g_per_scale).
    """
    # the setup cell, then the frog alone
    namespace = run_cells({}, SETUP7, notebook=NOTEBOOK7)
    namespace["PARAMS"]["test_images"] = []
    return run_cells(namespace, COMPUTE7, notebook=NOTEBOOK7)


# ── Notebook 8: stimulus features at each noise level ──────────────────────
NOTEBOOK8 = REPO / "notebooks" / "8_stimulus_features_per_scale.ipynb"
STIMULUS_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "stimulus_features"

# Lines of the code cells of notebook 8 (setup, stimulus features, plot); its functions
# are in figures/per_scale.py.
SETUP8 = '"prior_area": "V1",'
ENV8 = "env = compute_env("
PLOT8 = 'records = {"stimulus": env[name]}'


def stimulus_inputs(path: Path | None = None) -> tuple[dict, dict, dict]:
    """
    (examples, env, results) like those of notebook 8, for the frog only: env, the
    stimulus features, from their stored fixture, and results, M_t(x) of V1 and V4, from
    the fixture of notebook 7 (`per_scale_inputs`: the same frog and seeds).
    """
    from types import SimpleNamespace

    import numpy as np
    import torch

    # the frog and M_t(x), as notebook 7 stored them
    examples, results = per_scale_inputs()

    # the stored stimulus features, eigenvectors back to float32 (stored as float16)
    data = np.load(path or STIMULUS_FIXTURES / "stimulus_features_data.npz")
    record = SimpleNamespace(
        log_tau=torch.from_numpy(data["log_t"]),
        eigenvalues=torch.from_numpy(data["eigenvalues"]),
        eigenvectors=torch.from_numpy(data["eigenvectors"].astype(np.float32)),
    )
    return examples, {"frog": record}, results


def render_stimulus_plot(
    examples: dict, env: dict, results: dict, out_png: Path, figure_dir: Path
) -> dict:
    """
    Run notebook 8's plot cell on (examples, env, results), with the figures saved under
    `figure_dir`, and save the last figure drawn to `out_png` (100 dpi). Returns the
    namespace of the cell.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figures.per_scale import plot_per_scale

    # the names of the setup cell that the plot cell uses, and the settings of the plot
    plt.close("all")
    namespace = {"plt": plt, "Path": Path, "plot_per_scale": plot_per_scale}
    namespace.update(examples=examples, env=env, results=results)
    namespace["PARAMS"] = {
        "encoder": "inception",
        "areas": list(AREAS),
        "prior_area": "V1",
        "n_samples": 200,
        "figure_dir": str(figure_dir),
        "file_dpi": 100,
        "screen_dpi": 100,
        "show": ["frog"],
    }

    # the plot cell, then the figure of the last image
    run_cells(namespace, PLOT8, notebook=NOTEBOOK8)
    namespace["fig"].savefig(out_png, dpi=100)
    plt.close("all")
    return namespace


def run_notebook8_frog() -> dict:
    """
    Namespace after running notebook 8's setup and stimulus-features cells for the frog
    only (needs runs/, TinyImageNet and the multigeom with compute_env_per_scale).
    """
    # the setup cell, then the frog alone
    namespace = run_cells({}, SETUP8, notebook=NOTEBOOK8)
    namespace["PARAMS"]["test_images"] = []
    return run_cells(namespace, ENV8, notebook=NOTEBOOK8)


# ── Notebook 9: the envelope spectrum ──────────────────────────────────────
NOTEBOOK9 = REPO / "notebooks" / "9_envelope_spectrum.ipynb"
ENVELOPE_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "envelope_spectrum"

# Lines of the code cells of notebook 9 (setup, spectrum, Gaussian bound, plot); its
# functions are in figures/per_scale.py.
SETUP9 = '"prior_areas": ["V1", "V4"],'
SPECTRUM9 = "spectra[area] = compute_env_spectrum("
GAUSSIAN9 = "variances = compute_prior_variances("
PLOT9 = "plot_env_spectrum(spectra, gaussian, PARAMS)"


def envelope_inputs(path: Path | None = None) -> tuple[dict, "object", "object"]:
    """
    (spectra, variances, gaussian) like those of notebook 9, from the stored fixture:
    spectra maps each model to its curves (`log_tau` [J], `values` [images, J]),
    variances are those of the training images, and gaussian the Gaussian bound.
    """
    from types import SimpleNamespace

    import numpy as np
    import torch

    # the stored curves of each model, and the variances of the training images
    data = np.load(path or ENVELOPE_FIXTURES / "envelope_spectrum_data.npz")
    log_tau = torch.from_numpy(data["log_t"])
    spectra = {
        area: SimpleNamespace(
            log_tau=log_tau, values=torch.from_numpy(data[f"{area}_values"])
        )
        for area in AREAS
    }
    variances = torch.from_numpy(data["variances"])
    return spectra, variances, torch.from_numpy(data["gaussian"])


def render_envelope_plot(
    spectra: dict, variances, gaussian, out_png: Path, figure_dir: Path
) -> dict:
    """
    Run notebook 9's plot cell on (spectra, variances, gaussian), with the figure saved
    under `figure_dir`, and save it to `out_png` (100 dpi). Returns the namespace of the
    cell.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from figures.per_scale import plot_env_spectrum

    # the names of the setup cell that the plot cell uses, and the settings of the plot
    plt.close("all")
    namespace = {"Path": Path, "plot_env_spectrum": plot_env_spectrum}
    namespace.update(spectra=spectra, variances=variances, gaussian=gaussian)
    namespace["PARAMS"] = {
        "encoder": "inception",
        "n_noise": 8,
        "figure_dir": str(figure_dir),
        "file_dpi": 100,
        "screen_dpi": 100,
    }

    # the plot cell, then its figure
    run_cells(namespace, PLOT9, notebook=NOTEBOOK9)
    namespace["fig"].savefig(out_png, dpi=100)
    plt.close("all")
    return namespace


def run_notebook9() -> dict:
    """
    Namespace after running notebook 9's setup, spectrum and Gaussian-bound cells (needs
    runs/, TinyImageNet and the multigeom with compute_env_spectrum; about 2 min on a
    GPU).
    """
    return run_cells({}, SETUP9, SPECTRUM9, GAUSSIAN9, notebook=NOTEBOOK9)


# ── Notebook 10: the coding spectrum ───────────────────────────────────────
NOTEBOOK10 = REPO / "notebooks" / "10_coding_spectrum.ipynb"
CODING_FIXTURES = Path(__file__).resolve().parent / "fixtures" / "coding_spectrum"

# Lines of the code cells of notebook 10 (setup, spectrum, plot); its
# functions are in figures/per_scale.py.
SETUP10 = '"areas": ["V1", "V4"],'
SPECTRUM10 = "spectra[area] = compute_coding_spectrum("
PLOT10 = "plot_coding_spectrum(spectra, bootstrap, PARAMS)"

# the per-image terms of a coding spectrum, as multigeom.CodingSpectrumOutput holds them
CODING_TERMS = ("env", "coded", "gain", "sampled")


def coding_record(log_tau, env, coded, gain, sampled) -> "object":
    """
    A stand-in for multigeom.CodingSpectrumOutput (absent from the released package):
    the noise levels [J], the four terms of each image [images, J], and the three
    curves the notebook reads from them, averaged over the images.
    """
    from types import SimpleNamespace

    # the stored terms, and the curves: the errors without and with the responses (the
    # note's eq. 7), what the responses change, and notebook 7's estimate
    record = SimpleNamespace(
        log_tau=log_tau, env=env, coded=coded, gain=gain, sampled=sampled
    )
    record.rho = (env - coded).mean(dim=0)
    record.rho_gain = gain.mean(dim=0)
    record.rho_sampled = sampled.mean(dim=0)
    return record


def coding_inputs(path: Path | None = None) -> dict:
    """
    The spectra of notebook 10, from the stored fixture: each area mapped to its coding
    spectrum.
    """
    import numpy as np
    import torch

    # the stored terms of each area
    data = np.load(path or CODING_FIXTURES / "coding_spectrum_data.npz")
    log_tau = torch.from_numpy(data["log_t"])
    spectra = {}
    for area in AREAS:
        terms = {
            name: torch.from_numpy(data[f"{area}_{name}"]) for name in CODING_TERMS
        }
        spectra[area] = coding_record(log_tau, **terms)
    return spectra


def render_coding_plot(spectra: dict, out_png: Path, figure_dir: Path) -> dict:
    """
    Run notebook 10's plot cell on `spectra`, with the figure saved under
    `figure_dir`, and save it to `out_png` (100 dpi). Returns the namespace of the cell.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch

    from figures.per_scale import (
        compute_band,
        compute_bootstrap,
        compute_information,
        plot_coding_spectrum,
    )

    # the names of the setup cell that the plot cell uses, and the settings of the plot
    plt.close("all")
    namespace = {
        "Path": Path,
        "torch": torch,
        "compute_band": compute_band,
        "compute_bootstrap": compute_bootstrap,
        "compute_information": compute_information,
        "plot_coding_spectrum": plot_coding_spectrum,
        "spectra": spectra,
    }
    namespace["PARAMS"] = {
        "encoder": "inception",
        "n_noise": 8,
        "n_bootstrap": 2000,
        "bootstrap_seed": 0,
        "figure_dir": str(figure_dir),
        "file_dpi": 100,
        "screen_dpi": 100,
    }

    # the plot cell, then its figure
    run_cells(namespace, PLOT10, notebook=NOTEBOOK10)
    namespace["fig"].savefig(out_png, dpi=100)
    plt.close("all")
    return namespace


def run_notebook10() -> dict:
    """
    Namespace after running notebook 10's setup and spectrum cells
    (needs runs/, TinyImageNet and the multigeom with compute_coding_spectrum; about
    7 min on a GPU).
    """
    return run_cells({}, SETUP10, SPECTRUM10, notebook=NOTEBOOK10)
