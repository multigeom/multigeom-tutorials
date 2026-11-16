"""
Regenerate the fixtures of notebook 10 (the coding spectrum) from the trained
checkpoints and the data, by running the notebook's own cells (setup, spectrum and
fine-end limit; then the plot):

    tests/fixtures/coding_spectrum/coding_spectrum_data.npz
        tau = log t at the 31 noise levels; for the V1 and V4 models, the four terms of
        every test image (the errors without and with the responses, what the
        responses change, and notebook 7's estimate) and the trace of J(x) of every
        test image
    tests/fixtures/coding_spectrum/coding_spectrum_reference.png
        the notebook's figure rendered from that data

Needs runs/ (checkpoints, see README), the TinyImageNet data, a GPU (about 7 min), and a
multigeom with compute_coding_spectrum. Usage, from the repository root:

    python tests/fixtures/make_coding_spectrum_fixtures.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # tests (nb_helpers)

from nb_helpers import (  # noqa: E402
    CODING_FIXTURES,
    CODING_TERMS,
    coding_inputs,
    render_coding_plot,
    run_notebook10,
)


def get_fixture_data(namespace: dict) -> dict:
    """
    The arrays of the fixture, from the namespace of notebook 10 after its spectrum and
    fine-end-limit cells.
    """
    # the noise levels, then the four terms and the traces of J(x) of each area
    spectra = namespace["spectra"]
    data = {"log_t": next(iter(spectra.values())).log_tau.numpy()}
    for area, record in spectra.items():
        for name in CODING_TERMS:
            data[f"{area}_{name}"] = getattr(record, name).numpy()
        data[f"{area}_traces"] = namespace["traces"][area].numpy()
    return data


def main():
    # the notebook's computation, stored
    CODING_FIXTURES.mkdir(parents=True, exist_ok=True)
    data = get_fixture_data(run_notebook10())
    np.savez_compressed(CODING_FIXTURES / "coding_spectrum_data.npz", **data)

    # the notebook's figure, drawn from the stored data
    reference = CODING_FIXTURES / "coding_spectrum_reference.png"
    with tempfile.TemporaryDirectory() as tmp:
        render_coding_plot(*coding_inputs(), reference, Path(tmp))
    print("wrote", CODING_FIXTURES)


if __name__ == "__main__":
    main()
