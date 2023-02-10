from abc import ABC, abstractmethod
from typing import Any

from numpy import ndarray

from .config import Config
from .engine import NumpyEngine


class RiskModelConfig(Config):
    """
    Risk model configuration.

    Parameters
    ----------
    show_all_instruments : bool.
        Indicate whether to show all instruments. Default is False.
        If True, the instruments outside of the universe in each
        period may not be filtered out.
    """

    show_all_instruments: bool = False


class RiskModel(ABC):
    """
    Risk Model.

    The class is an abstract class to compute covariance and other
    related metrics.
    """

    ConfigClass = RiskModelConfig

    def __init__(
        self, engine: Any = None, show_all_instruments: bool = False, **kwargs
    ):
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
        self._config = self.ConfigClass(
            show_all_instruments=show_all_instruments, **kwargs
        )

    @abstractmethod
    def cov(self, **kwargs) -> ndarray:
        """
        Get the covariance matrix.

        Returns
        -------
        numpy.ndarray
            A square pairwise covariance matrix which its
            diagonal entries are the variances.
        """

    def corr(self, **kwargs):
        """
        Get the correlation matrix.

        Returns
        -------
        numpy.ndarray
            A square pairwise correlation matrix which its
            diagonal entries are all ones.
        """
        cov = self.cov(**kwargs)
        vol = self._engine.sqrt(self._engine.diagonal(cov))
        return ((cov / vol).T / vol).T

    def asdict(self):
        """
        Returns a dict representation of the object.
        """
        return self.config.dict()
