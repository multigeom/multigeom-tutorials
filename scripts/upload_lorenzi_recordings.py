# ruff: noqa: E501  (the dataset card's Markdown tables cannot be wrapped)
"""
Upload the Lorenzi retina recordings (one <cell>.npz per retinal ganglion cell) and
open-retina's cell metadata table to the Hugging Face dataset repository
steevelaquitaine/lorenzi-retina-recordings, with a dataset card. Notebook 22 downloads
them from there.

The recordings are B. Lorenzi's unpublished data: the repository is PRIVATE unless
--public is given, which must wait for his authorization.

Usage, from the repository root (logged in to Hugging Face):

    python scripts/upload_lorenzi_recordings.py            # private
    python scripts/upload_lorenzi_recordings.py --public   # after authorization only
"""

import sys
from collections import Counter
from pathlib import Path

from huggingface_hub import HfApi

REPO = "steevelaquitaine/lorenzi-retina-recordings"
DATA = Path(
    "/home/steeve/steeve/idv/datasets/proj_multigeom/open-retina-multigeom/"
    "lorenzi_adaptation/data"
)
METADATA = Path(
    "../open-retina-multigeom/openretina/data_io/lorenzi_adaptation/cell_metadata.csv"
)

CARD = """---
pretty_name: Lorenzi retina recordings (flashed natural images with adaptation)
license: other
license_name: unpublished-data
license_link: LICENSE
tags:
- neuroscience
- retina
- retinal-ganglion-cells
- spike-counts
---

# Lorenzi retina recordings

Responses of {n_cells} retinal ganglion cells to flashed natural images preceded by an
adapting image, recorded by Baptiste Lorenzi (Institut de la Vision). These are the
data the Lorenzi adaptation models (open-retina) were fitted to, and that the retina
encoders of [multigeom](https://github.com/multigeom/multigeom) use.

> **Unpublished data.** Do not use, share or publish without the authorization of
> Baptiste Lorenzi. Species, recording method and preparation: **to be completed by
> the data owner.**

## Files

- `data/<cell>.npz`: one file per cell, named `<cell number>-<experiment>` (e.g.
  `176-exp7`).
- `cell_metadata.csv`: one row per cell, from open-retina's
  `data_io/lorenzi_adaptation` (columns below).

## Stimulus and responses

Each clip shows an **adapting image for 16 frames (400 ms)**, then a **natural image
for 16 frames (400 ms)**, at 40 Hz. Responses are **spike counts in 32 bins of 25 ms**
from the adapting image's onset (bin 0 precedes the first clip frame, so frame f of a
clip aligns with bin f + 1).

| Key in `<cell>.npz` | Shape | Content |
|---|---|---|
| `frames` | [n_frames, H, W] float32 | the bank of stimulus frames (30 x 30 px for exp4 to exp6, 40 x 40 for exp7) |
| `grey_value` | scalar | the grey level of the stimulus |
| `train_psth` | [n_train_clips, 32] | spike counts of the training clips, **one trial each** (about 3,180 clips) |
| `train_adapt_idx`, `train_nat_idx` | [n_train_clips] | the frame (index into `frames`) of each training clip's adapting and natural image; the training clips all adapt to grey |
| `test_psth_by_trial` | [n_test_clips, 30, 32] | spike counts of the test clips, **30 repeats each** |
| `test_adapt_idx`, `test_nat_idx` | [n_test_clips] | the adapting and natural image of each test clip |

Test clips: exp4 and exp5, 3 adapting images x 10 natural images (30 clips); exp6 and
exp7, 6 adapting images x 5 natural images minus the 5 pairs where the adapting image
is the natural image itself (25 clips). Adapting image 1 is uniform grey.

## Experiments

| Experiment | Cells | Stimulus | µm per pixel |
|---|---|---|---|
{experiments}
## Cell types

From the experiments' rating sheets (`cell_type` in `cell_metadata.csv`; empty =
untyped):

| Type | Cells |
|---|---|
{types}
## `cell_metadata.csv`

`session` (the cell), `experiment`, `cell_type`, `test_reliability` (split-half noise
ceiling on the test set), `model_test_score` (the Lorenzi model's noise-corrected r²
on the test set), `stimulus_size_px`, `microns_per_px`, `n_grey_frames` (grey frames
before each clip at evaluation), `H_on`, `H_off` (fitted gain-control strengths), `A`
(fitted output scale).

## Loading

```python
from huggingface_hub import snapshot_download
from multigeom.datasets import load_lorenzi_cell

folder = snapshot_download("steevelaquitaine/lorenzi-retina-recordings",
                           repo_type="dataset", local_dir="data/lorenzi_adaptation")
cell = load_lorenzi_cell(f"{{folder}}/data", "176-exp7")
cell.test_counts.shape  # (25, 30, 32): clips, repeats, 25 ms bins
```

or directly with `numpy.load("data/176-exp7.npz")`.

## Citation

To be completed by the data owner (Lorenzi et al., in preparation).
"""

LICENSE = """Unpublished data of Baptiste Lorenzi (Institut de la Vision). All rights
reserved. Do not use, share or publish without his written authorization.
"""


def main(public: bool) -> None:
    """Upload the recordings, the metadata, the card and the licence."""
    import csv

    rows = list(csv.DictReader(METADATA.open()))
    n_cells = len(list(DATA.glob("*.npz")))
    per_experiment = Counter(r["experiment"] for r in rows)
    geometry = {
        r["experiment"]: (r["stimulus_size_px"], r["microns_per_px"]) for r in rows
    }
    experiments = "".join(
        f"| {e} | {per_experiment[e]} | {geometry[e][0]} x {geometry[e][0]} px"
        f" | {float(geometry[e][1]):.1f} |\n"
        for e in sorted(per_experiment)
    )
    types = Counter(r["cell_type"] or "untyped" for r in rows)
    order = sorted(types, key=lambda k: (k == "untyped", -types[k]))
    types_table = "".join(f"| {k} | {types[k]} |\n" for k in order)

    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=not public, exist_ok=True)
    if public:
        api.update_repo_settings(REPO, repo_type="dataset", private=False)
    card = CARD.format(n_cells=n_cells, experiments=experiments, types=types_table)
    for content, name in ((card, "README.md"), (LICENSE, "LICENSE")):
        api.upload_file(
            path_or_fileobj=content.encode(),
            path_in_repo=name,
            repo_id=REPO,
            repo_type="dataset",
        )
    api.upload_file(
        path_or_fileobj=str(METADATA),
        path_in_repo="cell_metadata.csv",
        repo_id=REPO,
        repo_type="dataset",
    )
    api.upload_folder(
        folder_path=DATA,
        path_in_repo="data",
        repo_id=REPO,
        repo_type="dataset",
        commit_message=f"The recordings of {n_cells} cells",
    )
    print(f"uploaded {n_cells} cells to {REPO} ({'public' if public else 'private'})")


if __name__ == "__main__":
    main(public="--public" in sys.argv[1:])
