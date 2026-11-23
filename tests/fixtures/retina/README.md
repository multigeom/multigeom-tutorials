# What Baptiste's retina figures show

## Comparing the figures by eye

Each of Baptiste's figures is stored here as `<name>_reference.png`. Every run of
`tests/test_retina.py` redraws ours next to it, as `<name>.png`, in the same layout:

| reference | ours |
|---|---|
| `test_retina_j_x_reference.png` (J(x)) | `test_retina_j_x.png` |
| `test_retina_g_x_reference.png` (G(x)) | `test_retina_g_x.png` |
| `test_retina_jx_gx_reference.png` (G(x) and J(x) on the bus) | `test_retina_jx_gx.png` |

Two differences are expected:
- **Signs:** the sign of an eigenvector is arbitrary, so red and blue can be swapped in
  a tile.
- **Order of J(x)'s eigenvectors:** eigenvalues a few % apart can swap places, so a tile
  may appear a column later in ours.

Baptiste's notes (below, from the `legend.md` that came with the figures) do not fully
say which cells, checkpoints or images each figure uses. These were found from the
figures' pixels and are used by `tests/test_retina.py`. All diffusion models are those
trained on TinyImageNet.

## What the tests need

- **The `openretina` package** from the private repository
  [steevelaquitaine/open-retina-multigeom](https://github.com/steevelaquitaine/open-retina-multigeom).
  It holds open-retina plus Baptiste's unpublished retina code, and only the files the
  tests load. Install it without its dependencies, so your torch is kept, then add what
  it imports:
  ```bash
  pip install --no-deps "git+https://github.com/steevelaquitaine/open-retina-multigeom.git"
  pip install jaxtyping hydra-core lightning tenacity opencv-python-headless seaborn moviepy
  ```
- **The checkpoints of the two cells**, in `runs/`, laid out as those of V1 and V4:
  `runs/encoders/lorenzi/<area>/best_model.pt` (the cell's Lorenzi model) and
  `runs/diffusion/lorenzi/<area>/final_model.pt` (the conditional diffusion model of its
  mosaic, trained on TinyImageNet). The areas are `retina-W3_176-exp7` and
  `retina-tOFFalpha_17-exp5`. They come from the Hugging Face repository
  `steevelaquitaine/multigeom`:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download("steevelaquitaine/multigeom", local_dir="runs", allow_patterns="*/lorenzi/*")
  ```
- **The TinyImageNet images**, in `data/tiny-imagenet-200/`.

## test_retina_j_x and test_retina_g_x

- **Cell:** W3 cell 176-exp7, the 80-cell mosaic, as Baptiste's notes say.
- **Images:** the 6 test images of open-retina's `run_lorenzi`. These are the TinyImageNet
  training images 50,000 to 50,005, in its seed-0 shuffled order, standardised together.
- **j_x:** J(x), the Poisson Fisher information. Each percentage is λ_k / Σ λ over all 80
  eigenvalues.
- **g_x:** G(x), with 150 samples per noise level. Each percentage is λ_k / Σ λ over the
  20 leading eigenvalues.

## test_retina_jx_gx

This is panel A of Baptiste's `test_retina1.png`, cropped. It has smaller tiles with
1-pixel frames, about 116 pixels for the image's 120. That size blurs the few-pixel
receptive fields of J(x) too much to compare them pixel by pixel. So the test checks
where each J(x) tile peaks: within 3 pixels of one of our eigenvectors, for e2–e5. The
G(x) rows are compared pixel by pixel, as in the other figures.

- **Image:** test image 2 of the 6 above (the bus).
- **Rows:**
  1. G(x) of W3 cell 176-exp7;
  2. G(x) of tOFFalpha cell 17-exp5;
  3. J(x) of tOFFalpha cell 17-exp5.
- **How the cell was found:** each of the 5 tOFFalpha cells was tried on the bus. J(x)
  does not depend on random draws, so its row identifies the cell exactly.
  - For 17-exp5, tiles e2–e5 correlate at ≥ 0.96 with our eigenvectors. The other cells
    reach at most 0.85.
  - Tile e1 is a receptive field in the tile's corner, partly under the printed share.
- **Check against the G(x) rows:**
  - W3 176-exp7: e1 0.94, e2 0.92, with shares 53/15/6/3/3 % (reference 53/15/6/3/2).
  - tOFFalpha 17-exp5: e1 0.98, e2 0.95, with shares 37/10/7/6/5 % (reference
    37/9/7/6/5).

The other panels of `test_retina1.png` (B–E) are not tested.

## Baptiste's notes

- **One cell to reproduce:** cell 176-exp7, TinyImageNet prior, Fisher information then
  multi-scale Fisher information (test_retina_j_x and test_retina_g_x).
- **Poisson noise:** Poisson rather than Gaussian response noise gives better
  eigenvectors on TinyImageNet; the bus image is his example. With Gaussian noise, the
  eigenvectors mostly followed the differences in mean brightness between images.
- **Panels C–E of test_retina1** (not kept here):
  - C and D compare the alignment of eigenvectors between cells of the same type with
    that between cells of different types, for G(x) (C) and J(x) (D). A metric that
    tells cell types apart gives a significantly lower second bar: it does for G(x), not
    for J(x).
  - E compares G(x) with J(x) within one cell type, using the same bars.
- **The alignment metric.** Neither subspace alignment nor the Bures-Wasserstein
  distance weights an eigenvector by its eigenvalue. Yet cell types differ mainly in the
  leading eigenvector, so weighting by the eigenvalues should widen the gap between
  same-type and different-type alignments. They ended up using the Bures-Wasserstein
  distance and showing that similarity decreases as more eigenvectors are added.
