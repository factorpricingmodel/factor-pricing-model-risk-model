from abc import ABC, abstractmethod
from typing import Any

from numpy import ndarray

from .engine import NumpyEngine


class RiskModel(ABC):
    """
    Risk Model.

    The class is an abstract class to compute covariance and other
    related metrics.
    """

    def __init__(self, engine: Any = None, show_all_instruments: bool = False):
        """
        Constructor.

        Parameters
        ----------
        engine : Engine object.
            Engine used in computation.

        show_all_instruments : bool.
            Indicate whether to show all instruments. Default is False.
            If True, the instruments outside of the universe in each
            period may not be filtered out.
        """
        self._engine = engine or NumpyEngine
        self._show_all_instruments = show_all_instruments

    @abstractmethod
    def cov(self) -> ndarray:
        """
        Get the covariance matrix.

        Returns
        -------
        numpy.ndarray
            A square pairwise covariance matrix which its
            diagonal entries are the variances.
        """

    def corr(self):
        """
        Get the correlation matrix.

        Returns
        -------
        numpy.ndarray
            A square pairwise correlation matrix which its
            diagonal entries are all ones.
        """
        cov = self.cov()
        vol = self._engine.sqrt(self._engine.diagonal(cov))
        return ((cov / vol).T / vol).T
