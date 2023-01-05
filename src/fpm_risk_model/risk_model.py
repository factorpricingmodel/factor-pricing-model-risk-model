from abc import ABC, abstractmethod
from typing import Any

from .engine import NumpyEngine


class RiskModel(ABC):
    """
    Risk Model.

    The class is an abstract class to compute covariance and other
    related metrics.
    """

    def __init__(self, engine: Any = None):
        """
        Constructor.

        Parameters
        ----------
        engine : Engine object.
            Engine used in computation.
        """
        self._engine = engine or NumpyEngine

    @abstractmethod
    def cov(self):
        """
        Get the covariance matrix.
        """

    def corr(self):
        """
        Get the correlation matrix.
        """
        cov = self.cov()
        vol = self._engine.sqrt(self._engine.diagonal(cov))
        return ((cov / vol).T / vol).T
