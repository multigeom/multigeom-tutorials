"""
Regenerate the fixtures of notebook 9 (the envelope spectrum) from the trained
checkpoints and the data, by running the notebook's own cells (setup, spectrum and
Gaussian bound; then the plot):

    tests/fixtures/envelope_spectrum/envelope_spectrum_data.npz
        tau = log t at the 31 noise levels, the curve of every test image for the V1
        and V4 models, the variances of the training images and the Gaussian bound
    tests/fixtures/envelope_spectrum/envelope_spectrum_reference.png
        the notebook's figure rendered from that data

Needs runs/ (checkpoints, see README), the TinyImageNet data, a GPU, and a multigeom
with compute_env_spectrum. Usage, from the repository root:

    python tests/fixtures/make_envelope_spectrum_fixtures.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests (nb_helpers)

from nb_helpers import (  # noqa: E402
    ENVELOPE_FIXTURES,
    envelope_inputs,
    render_envelope_plot,
    run_notebook9,
)


def get_fixture_data(namespace: dict) -> dict:
    """
    The arrays of the fixture, from the namespace of notebook 9 after its spectrum and
    Gaussian-bound cells.
    """
    # the noise levels, the curves of the images of each model
    spectra = namespace["spectra"]
    data = {"log_t": next(iter(spectra.values())).log_tau.numpy()}
    for area, record in spectra.items():
        data[f"{area}_values"] = record.values.numpy()

    # the variances of the training images, and the Gaussian bound
    data["variances"] = namespace["variances"].numpy()
    data["gaussian"] = namespace["gaussian"].numpy()
    return data


def main():
    # the notebook's computation, stored
    ENVELOPE_FIXTURES.mkdir(parents=True, exist_ok=True)
    data = get_fixture_data(run_notebook9())
    np.savez_compressed(ENVELOPE_FIXTURES / "envelope_spectrum_data.npz", **data)

    # the notebook's figure, drawn from the stored data
    reference = ENVELOPE_FIXTURES / "envelope_spectrum_reference.png"
    with tempfile.TemporaryDirectory() as tmp:
        render_envelope_plot(*envelope_inputs(), reference, Path(tmp))
    print("wrote", ENVELOPE_FIXTURES)


if __name__ == "__main__":
    main()
