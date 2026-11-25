"""
Upload the stored results of the notebooks (data/02_intermediate/) to the Hugging Face
dataset repository steevelaquitaine/multigeom-results, one folder per notebook, each
with a provenance.json (notebook, code versions, date), and a dataset card.

Usage, from the repository root (logged in to Hugging Face):

    python scripts/upload_intermediate.py              # every folder
    python scripts/upload_intermediate.py 20_retina_v2 # some folders

The notebooks download their folder from there when it is missing locally.
"""

import datetime
import json
import subprocess
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO = "steevelaquitaine/multigeom-results"
ROOT = Path("data/02_intermediate")

# the notebook that writes each folder
PRODUCERS = {
    "11_stimulus_features_and_fisher": "11_stimulus_features_and_fisher_per_cell",
    "12_retina_eigenvectors_of_G_per_scale": "12_retina_eigenvectors_of_G_per_scale",
    "13_retina_stimulus_features_per_scale": "13_retina_stimulus_features_per_scale",
    "14_retina_envelope_spectrum": "14_retina_envelope_spectrum",
    "15_retina_coding_spectrum": "15_retina_coding_spectrum",
    "16_whitening": "16_whitening_stimulus_features",
    "20_retina_v2": "20_retina_v2_cells_and_fisher",
    "21_posterior_samples": "21_posterior_samples",
    "retina": "12_retina_eigenvectors_of_G_per_scale (the retina image normalizer)",
}

CARD = """---
license: mit
pretty_name: multigeom intermediate results
---

# multigeom intermediate results

The stored results of the notebooks of
[multigeom-tutorials](https://github.com/multigeom/multigeom-tutorials) (branch
`steeve-g`), so that the notebooks redraw their figures in minutes instead of
recomputing them (up to several hours on a GPU). One folder per notebook, with the
layout of the tutorials' `data/02_intermediate/`; each has a `provenance.json`
(notebook, code versions, date). Read the files with
`multigeom.utils.load_intermediate`.

```python
from huggingface_hub import snapshot_download
snapshot_download(repo_id="steevelaquitaine/multigeom-results", repo_type="dataset",
                  local_dir="data/02_intermediate", allow_patterns=["20_retina_v2/*"])
```

| Folder | Notebook |
|---|---|
"""


def get_commit(path: str) -> str:
    """The short commit of the git repository at `path`, marked if it has changes."""
    commit = subprocess.check_output(
        ["git", "-C", path, "rev-parse", "--short", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(["git", "-C", path, "status", "--short"], text=True)
    return commit + (" (with uncommitted changes)" if dirty.strip() else "")


def main(folders: list) -> None:
    """Upload `folders` (all of PRODUCERS if empty), with their provenance and the
    dataset card."""
    api = HfApi()
    api.create_repo(REPO, repo_type="dataset", private=False, exist_ok=True)
    rows = "".join(f"| `{k}/` | {v} |\n" for k, v in PRODUCERS.items())
    api.upload_file(
        path_or_fileobj=(CARD + rows).encode(),
        path_in_repo="README.md",
        repo_id=REPO,
        repo_type="dataset",
        commit_message="Dataset card",
    )
    code = {
        "multigeom-tutorials": get_commit("."),
        "multigeom": get_commit("../multigeom"),
    }
    for folder in folders or PRODUCERS:
        provenance = {
            "notebook": PRODUCERS[folder],
            "code": code,
            "uploaded": datetime.date.today().isoformat(),
        }
        (ROOT / folder / "provenance.json").write_text(json.dumps(provenance, indent=2))
        api.upload_folder(
            folder_path=ROOT / folder,
            path_in_repo=folder,
            repo_id=REPO,
            repo_type="dataset",
            commit_message=f"Results of {PRODUCERS[folder].split(' ')[0]}",
        )
        print("uploaded", folder)


if __name__ == "__main__":
    main(sys.argv[1:])
