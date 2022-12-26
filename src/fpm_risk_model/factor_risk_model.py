from abc import ABC
from typing import Any, Dict, Optional, Union

from numpy import ndarray


class FactorRiskModel(ABC):
    """
    Factor Risk Model.

    The class is an abstract class to fit the factor risk model.

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
        factor_exposures: Optional[Union[ndarray, Dict[Any, ndarray]]] = None,
        factor_returns: Optional[Union[ndarray, Dict[Any, ndarray]]] = None,
        factor_covariances: Optional[Union[ndarray, Dict[Any, ndarray]]] = None,
        residual_returns: Optional[Union[ndarray, Dict[Any, ndarray]]] = None,
    ):
        """
        Constructor

        Parameters
        ----------
        factor_exposures : Optional[Union[ndarray, Dict[Any, ndarray]]
          Factor exposures of the factor risk model.
        factors_returns : Optional[Union[ndarray, Dict[Any, ndarray]]]
          Factor returns of the factor risk model.
        factor_covariances : Optional[Union[ndarray, Dict[Any, ndarray]]]
          Factor covariances of the factor risk model.
        residual_returns : Optional[Union[ndarray, Dict[Any, ndarray]]]
          Residual returns of the factor risk model.
        """
        self._factor_exposures = factor_exposures
        self._factor_returns = factor_returns
        self._factor_covariances = factor_covariances
        self._residual_returns = residual_returns

    @property
    def factor_exposures(self) -> Union[ndarray, Dict[Any, ndarray]]:
        """
        Return the factor exposures.

        Return
        ------
        Union[ndarray, Dict[Any, ndarray]]
          Matrix in dimension (n, N) where N is the number of
          instruments and n is the number of components in PCA,
          or a dictionary of which its values are matrices in
          the mentioned format.
        """
        return self._factor_exposures

    @property
    def factor_returns(self) -> Union[ndarray, Dict[Any, ndarray]]:
        """
        Return the factor returns.

        Return
        ------
        Union[ndarray, Dict[Any, ndarray]]
          Matrix in dimension (T, n) where n is the number of
          components in PCA and T is the number of time frames,
          or a dictionary of which its values are matrices in
          the mentioned format.
        """
        return self._factor_returns

    @property
    def factor_covariances(self) -> Union[ndarray, Dict[Any, ndarray]]:
        """
        Return the factor returns.

        Return
        ------
        Union[ndarray, Dict[Any, ndarray]]
          Matrix in dimension (n, n) where n is the number of
          components in PCA,
          or a dictionary of which its values are matrices in
          the mentioned format.
        """
        return self._factor_covariances

    @property
    def residual_returns(self) -> ndarray:
        """
        Return the residual returns.

        Return
        ------
        Union[ndarray, Dict[Any, ndarray]]
          Matrix in dimension (T, N) where N is the number of
          instruments and T is the number of time frames,
          or a dictionary of which its values are matrices in
          the mentioned format.
        """
        return self._residual_returns

    def fit(self, X: ndarray) -> object:
        """
        Fit the model.
        """
        pass

    def copy(self) -> object:
        """
        Copy the model.
        """
        return FactorRiskModel(
            factor_exposures=self._factor_exposures.copy(),
            factor_covariances=self._factor_covariances.copy(),
            factor_returns=self._factor_returns.copy(),
            residual_returns=self._residual_returns.copy(),
        )
