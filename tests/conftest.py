"""Settings shared by the tests: the notebooks need the multigeom package installed (pip install -e <multigeom>)."""
import os

# diffusers imports transformers, which imports TensorFlow/JAX when installed; that is slow and can hang.
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
