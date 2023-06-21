from abc import ABC, abstractmethod
from typing import Any, Union

from numpy import diagonal, ndarray, sqrt
from pandas import DataFrame, Series

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
        self._engine = engine or NumpyEngine()
        self._config = self.ConfigClass(
            show_all_instruments=show_all_instruments, **kwargs
        )

    @property
    def config(self):
        """
        Return the configuration object.
        """
        return self._config

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

    def vol(self, **kwargs) -> ndarray:
        """
        Get the volatility series.

        Returns
        -------
        numpy.ndarray
            Volatility series derived from covariance matrix.
        """
        cov = self.cov(**kwargs)
        vol = sqrt(diagonal(cov))
        if isinstance(cov, DataFrame):
            vol = Series(vol, index=cov.index)
        return vol

    def corr(self, **kwargs) -> ndarray:
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

    @staticmethod
    def _to_numpy(values: Union[ndarray, DataFrame]) -> ndarray:
        """
        Convert the values to a numpy array
        """
        if values is None:
            return values
        elif isinstance(values, (DataFrame, Series)):
            return values.values
        elif isinstance(values, ndarray):
            return values
        else:
            raise TypeError(
                "Expect either pandas DataFrame or numpy array, "
                f"but got {values.__class__.__name__}"
            )
