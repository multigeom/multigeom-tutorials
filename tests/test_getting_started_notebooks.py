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
NB2 = REPO / "notebooks" / "2_getting_started_diffusion_and_g.ipynb"


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


@pytest.fixture(scope="module")
def notebook1(tiny_data, tmp_path_factory):
    out = tmp_path_factory.mktemp("runs") / "getting_started"
    import os
    old = os.environ.get("NEURIPS_DATA_DIR")
    os.environ["NEURIPS_DATA_DIR"] = str(tiny_data)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        ns = run_all_code_cells({}, NB1, overrides={
            "N_NEURONS": "25", "PRETRAIN_IMAGES_PER_CLASS": "2", "PRETRAIN_EPOCHS": "1", "N_NEURAL_TRAIN": "60",
            "N_NEURAL_TEST": "40", "FIT_EPOCHS": "3", "N_FISHER_IMAGES": "2", "OUT_DIR": f"Path({str(out)!r})"})
        yield ns, out
    finally:
        os.chdir(cwd)
        if old is None:
            os.environ.pop("NEURIPS_DATA_DIR", None)
        else:
            os.environ["NEURIPS_DATA_DIR"] = old


def test_notebook_1_builds_fits_saves_and_reloads_an_encoder(notebook1):
    ns, out = notebook1
    assert (out / "encoder.pt").exists() and (out / "teacher.pt").exists()
    assert ns["same"] is True                                            # the reloaded encoder predicts the same
    assert ns["report"].n_train > 0 and np.isfinite(ns["r_clean"]) and np.isfinite(ns["r_noisy"])
    assert ns["reloaded"].metadata["noise_sigma"] == 0.22 and ns["reloaded"].n_neurons == 25
    assert ns["report_contract"].n_neurons == 25 and not ns["report_contract"].warnings


def test_notebook_1_compares_readouts_and_the_fisher_information_with_the_teacher(notebook1):
    ns, _ = notebook1
    assert set(ns["results"]) == {"linear, trained backbone", "gaussian, trained backbone", "factorized, trained backbone",
                                  "gaussian, untrained backbone"}
    assert ns["j_fit"].eigenvectors.shape == ns["j_true"].eigenvectors.shape == (20, 4096)
    assert len(ns["scores"]) == 2 and all(0 <= s <= 1 for s in ns["scores"])


def test_notebook_2_trains_a_diffusion_model_and_computes_g(notebook1, tiny_data):
    pytest.importorskip("diffusers")
    ns1, out = notebook1
    import os
    old = os.environ.get("NEURIPS_DATA_DIR")
    os.environ["NEURIPS_DATA_DIR"] = str(tiny_data)
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        ns = run_all_code_cells({}, NB2, overrides={
            "IMAGES_PER_CLASS": "1", "EPOCHS": "1", "N_MC": "2", "N_TEST_IMAGES": "2", "OUT_DIR": f"Path({str(out)!r})"})
    finally:
        os.chdir(cwd)
        if old is None:
            os.environ.pop("NEURIPS_DATA_DIR", None)
        else:
            os.environ["NEURIPS_DATA_DIR"] = old
    assert (out / "diffusion" / "final_model.pt").exists() and len(ns["history"]["loss"]) == 1
    assert ns["responses"].shape == (200, 25)
    assert ns["g"].eigenvectors.shape == (20, 4096) and torch.isfinite(ns["g"].eigenvalues).all()
    assert all(np.array(c).shape == (2, 20) for c in ns["curves"].values())
