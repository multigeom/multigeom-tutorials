"""
Settings shared by the tests: the notebooks need the multigeom package installed (pip
install -e <multigeom>).
"""

import os

import matplotlib
import pytest
import torch

# diffusers imports transformers, which imports TensorFlow/JAX when installed; that is
# slow and can hang.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")


@pytest.fixture(autouse=True)
def restore_global_settings():
    """
    Undo, after each test, the global settings a notebook cell may change: the torch
    default dtype (the Foundations notebooks set float64, and models of later tests
    then get float64 inputs) and the matplotlib rcParams (their figure style changes
    the plots compared with reference pictures). The backend is kept.
    """
    dtype = torch.get_default_dtype()
    with matplotlib.rc_context():
        yield
    torch.set_default_dtype(dtype)
