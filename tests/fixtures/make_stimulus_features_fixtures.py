"""
Regenerate the fixtures of notebook 8 (the stimulus features M_t^env(x) at each noise
level, next to M_t(x)) from the trained checkpoints, by running the notebook's own cells
(setup and stimulus features, for the frog only; then the plot):

    tests/fixtures/stimulus_features/stimulus_features_data.npz
        the frog's stimulus features: the settings (samples, noisy images per probe,
        finite-difference step), tau = log t at the 24 noise levels, and the
        eigenvalues and the leading 4 eigenvectors (float16) at each level
    tests/fixtures/stimulus_features/stimulus_features_reference.png
        the notebook's figure of the frog rendered from that data and from notebook 7's
        fixture of M_t(x) (tests/fixtures/per_scale/, regenerate it first if needed)

Needs runs/ (checkpoints, see README), the TinyImageNet data, a GPU, and a multigeom
with compute_env_per_scale. Usage, from the repository root:

    python tests/fixtures/make_stimulus_features_fixtures.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests (nb_helpers)

from nb_helpers import (  # noqa: E402
    STIMULUS_FIXTURES,
    render_stimulus_plot,
    run_notebook8_frog,
    stimulus_inputs,
)


def get_fixture_data(namespace: dict) -> dict:
    """
    The arrays of the fixture, from the namespace of notebook 8 after its stimulus
    features cell (frog only).
    """
    # the settings of the estimate, and the noise levels
    params, record = namespace["PARAMS"], namespace["env"]["frog"]
    data = {
        "n_samples": np.array(params["n_samples"]),
        "n_noisy": np.array(params["n_noisy"]),
        "eta": np.array(params["eta"]),
        "log_t": record.log_tau.numpy(),
    }

    # the eigenpairs, the eigenvectors in float16 to halve the file
    data["eigenvalues"] = record.eigenvalues.numpy()
    data["eigenvectors"] = record.eigenvectors.numpy().astype(np.float16)
    return data


def main():
    # the notebook's stimulus features of the frog, stored
    STIMULUS_FIXTURES.mkdir(parents=True, exist_ok=True)
    data = get_fixture_data(run_notebook8_frog())
    np.savez_compressed(STIMULUS_FIXTURES / "stimulus_features_data.npz", **data)

    # the notebook's figure of the frog, drawn from the stored data
    reference = STIMULUS_FIXTURES / "stimulus_features_reference.png"
    with tempfile.TemporaryDirectory() as tmp:
        render_stimulus_plot(*stimulus_inputs(), reference, Path(tmp))
    print("wrote", STIMULUS_FIXTURES)


if __name__ == "__main__":
    main()
