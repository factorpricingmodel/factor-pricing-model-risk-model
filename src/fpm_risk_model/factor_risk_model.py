from abc import ABC

from numpy import ndarray


class FactorRiskModel(ABC):
    """
    Factor Risk Model.

    The model contains factor exposures, factors, and
    residual returns.
    """

    def __init__(self):
        self._factor_exposures = None
        self._factors = None
        self._residual_returns = None

    @property
    def factor_exposures(self) -> ndarray:
        """
        Return the factor exposures.

        Return
        ------
        ndarray
          Matrix in dimension (N, n) where N is the number of
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
          Matrix in dimension (n, T) where n is the number of
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
          Matrix in dimension (N, T) where N is the number of
          instruments and T is the number of time frames.
        """
        return self._residual_returns
