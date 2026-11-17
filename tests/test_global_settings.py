"""
Regression test of the fixture `restore_global_settings` (conftest.py): the setup cell
of a Foundations notebook sets the torch default dtype to float64 and updates the
matplotlib rcParams. Without the fixture both stay changed for the tests that run after
it: the getting-started tests then failed with "expected scalar type Float but found
Double", and the plots compared with reference pictures changed. The first test runs
that cell, the second checks that the settings are back. Pytest runs the tests of a file
in order, so this holds whatever the order of the files.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from nb_helpers import REPO, run_cells  # noqa: E402

NB_F1 = REPO / "notebooks" / "f1_what_one_response_tells_you.ipynb"
BEFORE = {}  # settings before the setup cell, filled by the first test


def settings() -> tuple:
    """(torch default dtype, matplotlib rcParams without the backend)."""
    rc = {k: v for k, v in plt.rcParams.items() if k != "backend"}
    return torch.get_default_dtype(), rc


def test_a_foundations_setup_cell_changes_the_global_settings():
    BEFORE["settings"] = settings()
    run_cells({}, "torch.set_default_dtype", notebook=NB_F1)
    dtype, rc = settings()
    assert dtype == torch.float64
    assert rc != BEFORE["settings"][1]


def test_b_the_global_settings_are_restored_after_the_test():
    if "settings" not in BEFORE:
        pytest.skip("run the whole file: the first test sets BEFORE")
    dtype, rc = settings()
    assert dtype == BEFORE["settings"][0]
    assert rc == BEFORE["settings"][1]
