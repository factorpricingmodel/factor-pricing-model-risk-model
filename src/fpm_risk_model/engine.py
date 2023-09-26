from contextlib import contextmanager
from os import environ
from typing import Any

_BACKEND_ENGINE = "numpy"
_SUPPORTED_ENGINES = ["numpy", "tensorflow", "cupy", "jax", "torch"]


def backend():
    return _BACKEND_ENGINE


def set_backend(library_name):
    """
    Set backend engine.

    The function sets the backend engine in global level.

    Parameters
    ----------
    library_name : str
        Library name. Default is `numpy`. Options are `numpy`, `tensorflow`,
        `cupy`, `jax` and `torch`.
    """
    library_name = library_name.lower()
    if library_name not in _SUPPORTED_ENGINES:
        raise ValueError(
            "Only `numpy`, `tensorflow`, `cupy`, `jax` and `torch` are supported, "
            f"but not {library_name}"
        )
    global _BACKEND_ENGINE
    _BACKEND_ENGINE = library_name
    return _BACKEND_ENGINE


@contextmanager
def use_backend(library_name="numpy"):
    """
    NumPy engine selection.

    The function is a context manager to enable users to switch to a
    specific library as a replacement of NumPy in CPU.

    Parameters
    ----------
    library_name : str
        Library name. Default is `numpy`. Options are `numpy`, `tensorflow`,
        `cupy`, `jax` and `torch`.
    """
    library_name = library_name.lower()
    if library_name not in _SUPPORTED_ENGINES:
        raise ValueError(
            "Only `numpy`, `tensorflow`, `cupy`, `jax` and `torch` are supported, "
            f"but not {library_name}"
        )
    global _BACKEND_ENGINE
    _original = _BACKEND_ENGINE
    try:
        _BACKEND_ENGINE = library_name
        yield
    finally:
        _BACKEND_ENGINE = _original


class NumpyEngine:
    """
    NumPy engine.
    """

    def __getattribute__(self, __name: str) -> Any:
        if __name == "name":
            return _BACKEND_ENGINE

        try:
            if _BACKEND_ENGINE == "numpy":
                import numpy as anp
            elif _BACKEND_ENGINE == "tensorflow":
                import tensorflow.experimental.numpy as anp

                anp.experimental_enable_numpy_behavior()
            elif _BACKEND_ENGINE == "cupy":
                import cupy as anp
            elif _BACKEND_ENGINE == "jax":
                import jax.numpy as anp
            elif _BACKEND_ENGINE == "torch":
                import torch as anp

                anp.array = anp.tensor
                anp.ndarray = anp.Tensor
                anp.newaxis = None
            else:
                raise ValueError(f"Cannot recognize backend {_BACKEND_ENGINE}")
        except ImportError:
            raise ImportError(
                "Library `numpy` cannot be imported from backend engine "
                f"{_BACKEND_ENGINE}. Please make sure to install the library "
                f"via `pip install {_BACKEND_ENGINE}`."
            )

        try:
            return getattr(anp, __name)
        except AttributeError:
            raise AttributeError(
                f"Cannot get attribute / function ({__name}) from numpy library in "
                f"backend engine {_BACKEND_ENGINE}"
            )


class LinAlgEngine:
    """
    Linear algebra engine.
    """

    def __getattribute__(self, __name: str) -> Any:
        if __name == "name":
            return _BACKEND_ENGINE

        try:
            if _BACKEND_ENGINE == "numpy":
                import numpy.linalg as alinalg
            elif _BACKEND_ENGINE == "tensorflow":
                import tensorflow.experimental.numpy as anp
                import tensorflow.linalg as alinalg

                anp.experimental_enable_numpy_behavior()
            elif _BACKEND_ENGINE == "cupy":
                import cupy.linalg as alinalg
            elif _BACKEND_ENGINE == "jax":
                import jax.numpy.linalg as alinalg
            elif _BACKEND_ENGINE == "torch":
                import torch.linalg as alinalg
            else:
                raise ValueError(f"Cannot recognize backend {_BACKEND_ENGINE}")
        except ImportError:
            raise ImportError(
                "Library `linalg` cannot be imported from backend engine "
                f"{_BACKEND_ENGINE}. Please make sure to install the library "
                f"via `pip install {_BACKEND_ENGINE}`."
            )

        try:
            return getattr(alinalg, __name)
        except AttributeError:
            raise AttributeError(
                f"Cannot get attribute / function ({__name}) from linalg library in "
                f"backend engine {_BACKEND_ENGINE}"
            )


set_backend(environ.get("FPM_BACKEND_ENGINE", "numpy"))
numpy = NumpyEngine()
linalg = LinAlgEngine()
