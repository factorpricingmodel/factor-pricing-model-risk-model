from typing import Any

import numpy


class NumpyEngine:
    """
    Numpy Engine.
    """

    def __getattribute__(self, __name: str) -> Any:
        return getattr(numpy, __name)
