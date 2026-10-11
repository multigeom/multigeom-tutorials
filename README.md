# multigeom-tutorials

**Landing page: <https://multigeom.github.io>**

Tutorial notebooks for [`multigeom`](https://github.com/sazio/MultiscaleInformationGeometry), the Python package that
estimates the multi-scale Fisher information geometry of neural population codes with conditional diffusion models
(paper: *A multi-scale information geometry reveals the structure of mutual information in neural populations*).

The package holds the code. This repository holds what a user needs to see it work on real use cases: five notebooks that
bring your own encoder or reproduce figures of the paper, the data download scripts, and the tests that run the notebooks
and compare their figures with stored references. It is separate from the package so that notebook outputs, which are
large and unreadable in diffs, do not clutter the package history.

## Notebooks

| Notebook | What it does | Needs | Time |
|---|---|---|---|
| `0_bring_your_own_encoder` | Checks that your own PyTorch model meets the encoder contract and computes its Fisher metric J(x). | the package only, CPU | seconds |
| `1_leading_eigenvectors_of_G` | Leading eigenvectors of G(x) and J(x) for an example image (paper Fig. 3D-F, 4B-C). | data, checkpoints, GPU | about 2 min |
| `2_subspace_alignment` | G(x) and J(x) for many images, split over local GPUs or Modal, then the subspace alignment (Fig. 4D-E). | data, checkpoints, GPU | 1 min to hours |
| `3_local_metric_field` | Local metric field around a reference image (Fig. 5). | data, checkpoints, GPU | about 25 min |
| `4_bures_wasserstein` | Bures-Wasserstein distance between geometries (Fig. 6). | the outputs of notebook 2 | minutes, CPU |

Start with notebook 0: it runs anywhere. Notebooks 1 to 4 reproduce the paper's analyses and need the data and checkpoints
below.

## Installation

Python 3.11. The notebooks need a GPU, except notebooks 0 and 4. Run these commands from the root of this repository.

1. **Create the environment** with conda. `environment.yml` pins PyTorch 2.7 for CUDA 12.8; edit the `--extra-index-url`
   and the torch lines for another CUDA version.

   ```bash
   conda env create -f environment.yml --prefix ./env     # about 1 min
   conda activate ./env
   ```

   On a machine without a GPU, use `environment-ci.yml` (CPU-only PyTorch) the same way.

2. **Install the package** into that environment.

   ```bash
   pip install "multigeom[all]"
   ```

   The `all` extra adds the reference encoders, the fitting code, the figures and Modal. Until `multigeom` is released on
   PyPI, install it from a clone instead:

   ```bash
   git clone https://github.com/sazio/MultiscaleInformationGeometry.git ../MultiscaleInformationGeometry
   pip install -e "../MultiscaleInformationGeometry[all]"
   ```

3. **Create the Jupyter kernel** and open the notebooks with it.

   ```bash
   python -m ipykernel install --user --name multigeom --display-name "multigeom"
   jupyter lab notebooks/
   ```

4. **Check the installation.**

   ```bash
   python -c "import multigeom; print(multigeom.__version__)"
   ```

Every notebook finds the repository root (the folder with `environment.yml`) and works from there, so `data/` and `runs/`
are read from this folder whatever the notebook's own location.

## Data and checkpoints (notebooks 1 to 4)

Notebook 0 needs neither. The other notebooks read:

- `data/`: TVSD, THINGS images and TinyImageNet; `data/README.md` describes each and the expected layout. The scripts `notebooks/download_tsvd_data.sh`,
  `notebooks/download_things_images.sh` and `notebooks/download_tiny_imagenet.sh` fetch them; notebook 1 describes the layout.
  Set `NEURIPS_DATA_DIR=/path/to/storage` to read the datasets from elsewhere.
- `runs/`: the trained encoders and diffusion models.

  ```bash
  huggingface-cli download steevelaquitaine/multigeom --local-dir runs
  ```

Both folders are ignored by git (except `data/README.md`). If you already have them elsewhere, link them (`ln -s <path>/runs runs`).

## Tests

The tests run the notebooks' cells and compare the figures they draw with the references in `tests/fixtures/`.

```bash
pytest                                           # everything; tests that need data or checkpoints skip themselves without them
pytest -m "not slow and not gpu and not pretrained"
```

The reference figures can be regenerated with the scripts in `tests/fixtures/`.

## Layout

```
notebooks/    the notebooks and the dataset download scripts
tests/        tests of the notebooks; fixtures/ holds the reference figures
environment.yml      GPU environment          environment-ci.yml   CPU environment (tests)
```

## License

MIT, like the package. The datasets keep their own licenses.
