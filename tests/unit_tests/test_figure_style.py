"""
Check the helpers of `figures/style.py` that set the plotting backend and read the
scores of the encoders.

Unit tests of the helpers of figures/style.py on inputs whose answer is known: no data,
checkpoints or GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch


def test_style_keeps_the_notebook_backend_in_a_jupyter_kernel(monkeypatch):
    """
    Verify that importing `figures.style` in a Jupyter kernel keeps the notebook
    backend, so figures show as pictures (it switches to Agg only in scripts).

    In a kernel, importing the figure style must not switch to Agg (a figure would
    show as text, not a picture).
    """
    import sys
    import types

    from figures import style

    fake = types.ModuleType("IPython")
    fake.get_ipython = lambda: types.SimpleNamespace(kernel=object())
    monkeypatch.setitem(sys.modules, "IPython", fake)
    assert style._in_notebook() is True
    fake.get_ipython = lambda: None
    assert style._in_notebook() is False
    monkeypatch.delitem(sys.modules, "IPython")
    monkeypatch.setitem(sys.modules, "IPython", None)  # not installed
    assert style._in_notebook() is False


@pytest.mark.parametrize("layout", ["older paper checkpoint", "NeuralEncoder.save"])
def test_encoder_checkpoint_gives_the_scores_in_both_layouts(tmp_path, layout):
    """
    Verify that `encoder_checkpoint` reads the test scores of an encoder checkpoint in
    both layouts (older paper checkpoints and `NeuralEncoder.save`), and returns None
    when there is none.
    """
    from figures.style import encoder_checkpoint

    scores = {
        "per_neuron_test_corr": np.array([0.5, 0.6]),
        "oracle": np.array([0.9, 0.9]),
        "test_corr": 0.55,
    }
    ckpt = {"model_state_dict": {"readout.mu": torch.zeros(2, 2)}}
    ckpt.update(
        scores
        if layout == "older paper checkpoint"
        else {"metadata": {"area": "V1", **scores}}
    )
    d = tmp_path / "encoders" / "inception" / "V1"
    d.mkdir(parents=True)
    torch.save(ckpt, d / "best_model.pt")
    got = encoder_checkpoint(tmp_path, "inception", "V1", "test")
    np.testing.assert_array_equal(
        got["per_neuron_test_corr"], scores["per_neuron_test_corr"]
    )
    assert got["test_corr"] == 0.55 and "model_state_dict" in got
    assert encoder_checkpoint(tmp_path, "inception", "V4", "test") is None
