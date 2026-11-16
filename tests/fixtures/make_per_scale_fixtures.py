"""
Regenerate the fixtures of notebook 7 (the leading eigenvectors of M_t(x), the integrand
of G(x), at each noise level) from the trained checkpoints, by running the notebook's
own cells (setup, functions and compute, for the frog only; then the plot):

    tests/fixtures/per_scale/per_scale_data.npz
        the frog, its seed, the number of samples per noise level, tau = log t at the
        24 noise levels, and for V1 and V4 the eigenvalues and the leading 4
        eigenvectors (float16) at each level
    tests/fixtures/per_scale/per_scale_reference.png
        the notebook's figure of the frog rendered from that data

Needs runs/ (checkpoints, see README), the TinyImageNet data, a GPU, and a multigeom
with compute_g_per_scale. Usage, from the repository root:

    python tests/fixtures/make_per_scale_fixtures.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests (nb_helpers)

from nb_helpers import (  # noqa: E402
    AREAS,
    PER_SCALE_FIXTURES,
    per_scale_inputs,
    render_per_scale_plot,
    run_notebook7_frog,
)


def get_fixture_data(namespace: dict) -> dict:
    """
    The arrays of the fixture, from the namespace of notebook 7 after its compute cell
    (frog only).
    """
    # the frog, its seed, the samples per level and the noise levels
    image, seed = namespace["examples"]["frog"]
    first = namespace["results"][AREAS[0]]["frog"]
    data = {
        "image": image.numpy().reshape(64, 64),
        "seed": np.array(seed),
        "n_samples": np.array(namespace["PARAMS"]["n_samples"]),
        "log_t": first.log_tau.numpy(),
    }

    # the eigenpairs of each area, the eigenvectors in float16 to halve the file
    for area in AREAS:
        record = namespace["results"][area]["frog"]
        data[f"{area}_eigenvalues"] = record.eigenvalues.numpy()
        data[f"{area}_eigenvectors"] = record.eigenvectors.numpy().astype(np.float16)
    return data


def main():
    # the notebook's computation for the frog, stored
    PER_SCALE_FIXTURES.mkdir(parents=True, exist_ok=True)
    data = get_fixture_data(run_notebook7_frog())
    np.savez_compressed(PER_SCALE_FIXTURES / "per_scale_data.npz", **data)

    # the notebook's figure of the frog, drawn from the stored data
    examples, results = per_scale_inputs()
    reference = PER_SCALE_FIXTURES / "per_scale_reference.png"
    with tempfile.TemporaryDirectory() as tmp:
        render_per_scale_plot(examples, results, reference, Path(tmp))
    print("wrote", PER_SCALE_FIXTURES)


if __name__ == "__main__":
    main()
