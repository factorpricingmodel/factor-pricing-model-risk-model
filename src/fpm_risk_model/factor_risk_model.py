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

    The factor returns are returns among the date / time series for each
    factor.

    The residual returns are the idiosyncratic returns of the instruments
    regarding the specified factor exposures and returns.
    """

    def __init__(
        self,
        factor_exposures=None,
        factors_returns=None,
        residual_returns=None,
    ):
        self._factor_exposures = factor_exposures
        self._factors_returns = factors_returns
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
    def factor_returns(self) -> ndarray:
        """
        Return the factor returns.

        Return
        ------
        ndarray
          Matrix in dimension (T, n) where n is the number of
          components in PCA and T is the number of time frames.
        """
        return self._factor_returns

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
