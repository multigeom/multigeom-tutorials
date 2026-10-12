"""
Integration tests of the getting-started notebooks 1 and 2 (build your own encoder on synthetic neural data; train the
diffusion model and compute G(x)): their cells run end to end, on CPU, on a tiny fake TinyImageNet and with small sizes.
No dataset, checkpoint or network is needed, so these tests run on GitHub.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_helpers import REPO, run_all_code_cells  # noqa: E402

NB1 = REPO / "notebooks" / "1_getting_started_synthetic_neural_data.ipynb"
NB2 = REPO / "notebooks" / "2_getting_started_diffusion_and_pipeline.ipynb"


@pytest.fixture(scope="module")
def tiny_data(tmp_path_factory):
    """A fake TinyImageNet: 200 classes x 3 training images, 100 validation images (smooth random pictures)."""
    root = tmp_path_factory.mktemp("data")
    ti = root / "tiny-imagenet-200"
    ti.mkdir()
    wnids = [f"n{i:08d}" for i in range(200)]
    (ti / "wnids.txt").write_text("\n".join(wnids) + "\n")
    g = np.random.default_rng(0)

    def picture():
        low = g.random((8, 8, 3))
        big = np.kron(low, np.ones((8, 8, 1)))
        return Image.fromarray((big * 255).astype(np.uint8))

    for w in wnids:
        d = ti / "train" / w / "images"
        d.mkdir(parents=True)
        for k in range(3):
            picture().save(d / f"{w}_{k}.JPEG")
    vd = ti / "val" / "images"
    vd.mkdir(parents=True)
    lines = []
    for i in range(100):
        picture().save(vd / f"val_{i}.JPEG")
        lines.append(f"val_{i}.JPEG\t{wnids[i]}\t0\t0\t63\t63")
    (ti / "val" / "val_annotations.txt").write_text("\n".join(lines) + "\n")
    return root


def _in(tiny_data, notebook, overrides):
    """Run all the cells of `notebook` at the repository root, on the fake dataset; returns its namespace."""
    import os
    old = os.environ.get("NEURIPS_DATA_DIR")
    os.environ["NEURIPS_DATA_DIR"] = str(tiny_data)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        return run_all_code_cells({}, notebook, overrides=overrides)
    finally:
        os.chdir(cwd)
        if old is None:
            os.environ.pop("NEURIPS_DATA_DIR", None)
        else:
            os.environ["NEURIPS_DATA_DIR"] = old


@pytest.fixture(scope="module")
def notebook1(tiny_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "getting_started"
    ns = _in(tiny_data, NB1, {
        "N_NEURONS": "10", "N_TRAIN": "60", "N_TEST": "40", "IMAGES_PER_CLASS": "3", "PRETRAIN_EPOCHS": "1",
        "FIT_EPOCHS": "3", "N_FISHER_IMAGES": "2", "SAVE_DIR": f"Path({str(out)!r})"})
    return ns, out


def test_notebook_1_builds_fits_saves_and_reloads_an_encoder(notebook1):
    ns, out = notebook1
    assert (out / "encoder.pt").exists() and (out / "teacher.pt").exists()
    assert ns["same"] is True                                            # the reloaded encoder predicts the same
    assert ns["reloaded"].n_neurons == 10
    assert ns["report"].n_neurons == 10 and not ns["report"].warnings   # the encoder follows the contract


def test_notebook_1_compares_readouts_and_the_fisher_information_with_the_teacher(notebook1):
    ns, _ = notebook1
    assert set(ns["fits"]) == {"linear", "gaussian", "factorized", "gaussian, untrained backbone"}
    assert ns["j_fit"].eigenvectors.shape == ns["j_true"].eigenvectors.shape == (20, 4096) or \
        ns["j_fit"].eigenvectors.shape[1] == 4096
    assert set(ns["scores"]) == {1, 5, 10} and all(len(v) == 2 and all(0 <= s <= 1 for s in v) for v in ns["scores"].values())


def test_notebook_2_trains_a_diffusion_model_and_computes_g(notebook1, tiny_data):
    pytest.importorskip("diffusers")
    _, out = notebook1
    ns = _in(tiny_data, NB2, {
        "IMAGES_PER_CLASS": "1", "EPOCHS": "1", "N_MC": "4", "TIMESTEP_STEP": "160", "N_COMPARE": "2",
        "SAVE_DIR": f"Path({str(out)!r})"})
    assert (out / "diffusion" / "final_model.pt").exists() and len(ns["history"]["loss"]) == 1
    assert ns["responses"].shape == (200, 10)
    assert len(ns["G"]) == 2 and all(g.eigenvectors.shape[1] == 4096 and torch.isfinite(g.eigenvalues).all() for g in ns["G"])
