# multigeom-tutorials

**Landing page: <https://multigeom.github.io>**

Tutorial notebooks for [`multigeom`](https://github.com/multigeom/multigeom), the Python package that
estimates the multi-scale Fisher information geometry of neural population codes with conditional diffusion models
(paper: *A multi-scale information geometry reveals the structure of mutual information in neural populations*).

The package holds the code. This repository holds what a user needs to see it work: the notebooks, the data download
scripts, and the tests that run the notebooks and compare their figures with stored references. The notebooks form three
groups. Foundations computes each quantity on one neuron and a one-dimensional stimulus. Getting started applies the
package to encoders of images. Reproduce the paper redraws the figures of the paper. The repository is separate from the package so that notebook outputs, which are
large and unreadable in diffs, do not clutter the package history.

## Notebooks

**Foundations** (the quantities, on one neuron and a one-dimensional stimulus):

The stimulus x lies on [-3, 3] with a uniform prior. A neuron responds with its tuning curve plus Gaussian noise of
standard deviation 0.1. The bell-shaped neuron has the tuning curve exp(-x²/2). Its monotonic twin follows the same curve
up to the peak at x = 0 and then keeps rising to 2. The low-gain neuron is the monotonic twin at half its height. Every
quantity is computed exactly on a grid by `multigeom.lowdim`, without a diffusion model or a dataset.

| Notebook | What it does | Needs | Time |
|---|---|---|---|
| `f1_what_one_response_tells_you` | Build the response distribution of the bell-shaped neuron from repeats and the posterior of the stimulus given one response, with `response_density` and `posterior`. | the package, CPU | about 2 s |
| `f2_entropy_and_mutual_information` | Entropy of the prior and of a posterior with `entropy`, the bits removed by one response, and the mutual information of the three neurons with `mutual_information`. | the package, CPU | about 1 s |
| `f3_fisher_information` | How well two nearby stimuli can be told apart, the Fisher information J(x) with `compute_fisher`, the score, the Cramér-Rao bound checked by simulation, and J of eight neurons. | the package, CPU | about 1 s |
| `f4_from_fisher_to_geometry` | The Fisher distance between stimuli with `cumulative_distance` and `distance`, the 2 x 2 Fisher matrix of two neurons with `fisher_from_jacobian`, and three neurons whose Fisher distances do not order them by their mutual information. | the package, CPU | about 1 s |
| `f5_noise_as_a_ruler` | Blur a stimulus drawn from two narrow bumps by adding Gaussian noise of variance t (the blur level), compute the best guess of the stimulus from its blurred version, and split the information that survives the blur over blur levels, in closed form. | the package, CPU | about 2 s |
| `f6_which_stimuli_carry_the_information` | The Fisher information of the code seen through blur with `blurred_code`, the local information i(x) (the mutual information split over stimuli) with `compute_g`, and a test of locality against `specific_surprise` and `stimulus_specific_information`. | the package, CPU | about 9 s |
| `f7_multiscale_fisher_information` | The multi-scale Fisher information G(x), the Fisher information of the blurred code integrated over blur levels, of the three neurons with `compute_g`, the contribution of each blur level, and G distances next to Fisher distances. | the package, CPU | about 6 s |
| `f8_geometry_gives_back_the_information` | Check that half the prior average of the trace of G is the mutual information, on the three neurons, on Gaussian stimuli with a linear code with `gaussian_g` and `gaussian_information`, and on samples with `nearest_neighbour_information`. | the package, CPU | about 12 s |

**Getting started** (the basic blocks of the package):

| Notebook | What it does | Needs | Time |
|---|---|---|---|
| `0_use_a_builtin_encoder` | Load an encoder of the paper (fitted to macaque V1 or V4), look inside, predict neural responses, compute its Fisher metric J(x). | the package, internet, CPU | about 1 min |
| `1_create_your_encoder` | Build an encoder from scratch on synthetic neural data with a known ground truth: backbone, readout, `fit_readout`, `save`, `load`, `predict`. | TinyImageNet | about 1 min on a GPU |
| `2_compute_g` | Train the diffusion model of that encoder and compute the multi-scale metric G(x) next to J(x). | TinyImageNet | about 2 min on a GPU |

**Reproduce the paper**:

| Notebook | What it does | Needs | Time |
|---|---|---|---|
| `3_leading_eigenvectors_of_G` | Leading eigenvectors of G(x) and J(x) for an example image (paper Fig. 3D-F, 4B-C). | TinyImageNet, checkpoints, GPU | about 2 min |
| `4_subspace_alignment` | G(x) and J(x) for many images, split over local GPUs or Modal, then the subspace alignment (Fig. 4D-E). | TinyImageNet, checkpoints, GPU | 1 min to hours |
| `5_local_metric_field` | Local metric field around a reference image (Fig. 5). | TinyImageNet, checkpoints, GPU | about 25 min |
| `6_bures_wasserstein` | Bures-Wasserstein distance between geometries (Fig. 6). | the outputs of notebook 4 | minutes, CPU |

If the posterior, the mutual information and the Fisher information are new to you, start with notebook f1 and go on in
order to f8. To work with images, start with notebook 0. Notebooks 1 and 2 need only TinyImageNet. Notebooks 3 to 6
reproduce the paper's analyses and need the trained checkpoints and TinyImageNet, which the notebooks download (see
[Data and checkpoints](#data-and-checkpoints-notebooks-1-to-6)).

## Installation

Python 3.11. The Foundations notebooks and notebooks 0 and 6 run on a CPU. A GPU is recommended for notebooks 1 to 5
(notebooks 1 and 2 also run on a CPU, more slowly). Run these commands from the root of this repository.

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

   The `all` extra adds what reproducing the paper needs (the ON-OFF encoder and the TVSD loaders), Jupyter and Modal. Until `multigeom` is released on
   PyPI, install it from a clone instead:

   ```bash
   git clone https://github.com/multigeom/multigeom.git ../multigeom
   pip install -e "../multigeom[all]"
   ```

   The Foundations notebooks need a `multigeom` version that contains `multigeom.lowdim`. It is on the `develop` branch
   of the package and enters PyPI with the release after 0.1.0.dev4. Until then, install it from a clone as shown, after
   `git -C ../multigeom checkout develop`, or with
   `pip install "multigeom @ git+https://github.com/multigeom/multigeom.git@develop"`.

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

## Data and checkpoints (notebooks 1 to 6)

Notebook 0 and the Foundations notebooks need neither. Notebooks 1 to 5 need TinyImageNet, and notebooks 3 to 5 the
trained checkpoints. The notebooks download both:

- `data/`: TinyImageNet, with `multigeom.datasets.download_tinyimagenet()`.
  Set `NEURIPS_DATA_DIR=/path/to/storage` to read the datasets from elsewhere.
  TVSD and the THINGS images (`download_tvsd`, `download_things_images`) are only needed
  to refit the encoders; `data/README.md` describes them.
- `runs/`: the trained encoders and diffusion models, with
  `huggingface_hub.snapshot_download(repo_id="sazio/multigeom", local_dir="runs")`.

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
