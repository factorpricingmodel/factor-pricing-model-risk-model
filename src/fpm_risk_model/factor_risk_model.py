from abc import ABC, abstractmethod

from numpy import ndarray


class FactorRiskModel(ABC):
    """
    Factor Risk Model.

    The class is an abstract class to fit the factor risk model,
    and transform the input returns into a new model.

    The factor risk model contains the data attribute `factor_exposures`,
    `factors` and `residual_returns`.

    The factor exposures are the exposures of each instrument to the
    specified factors.

    The factors are the factor returns among the date / time series.

    The residual returns are the idiosyncratic returns of the instruments
    regarding the specified factor exposures and returns.
    """

    def __init__(
        self,
        factor_exposures=None,
        factors=None,
        residual_returns=None,
    ):
        self._factor_exposures = factor_exposures
        self._factors = factors
        self._residual_returns = residual_returns

    @property
    def factor_exposures(self) -> ndarray:
        """
        Return the factor exposures.

        Return
        ------
        ndarray
          Matrix in dimension (n, N) where N is the number of
          instruments and n is the number of components in PCA.
        """
        return self._factor_exposures

    @property
    def factors(self) -> ndarray:
        """
        Return the factors.

        Return
        ------
        ndarray
          Matrix in dimension (T, n) where n is the number of
          components in PCA and T is the number of time frames.
        """
        return self._factors

    @property
    def residual_returns(self) -> ndarray:
        """
        Return the residual returns.

        Return
        ------
        ndarray
          Matrix in dimension (T, N) where N is the number of
          instruments and T is the number of time frames.
        """
        return self._residual_returns

    @abstractmethod
    def fit(self, X: ndarray) -> object:
        """
        Fit the model.
        """
        pass
