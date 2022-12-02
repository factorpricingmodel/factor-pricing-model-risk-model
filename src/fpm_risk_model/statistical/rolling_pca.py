from typing import Optional, Union

import numpy as np
import pandas as pd

from ..factor_risk_model import FactorRiskModel
from .pca import PCA


class RollingPCA(FactorRiskModel):
    def __init__(
        self,
        n_components: int,
        rolling_timeframe: int,
        demean: Optional[bool] = True,
        speedup: Optional[bool] = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_components : int
          Number of components.
        rolling_timeframe: int
          Number of rolling time frames.
        demean : Optional[bool]
          Indicate whether to demean before fitting. Default is True.
        speedup: Optional[bool]
          Indicate whether to speed up the computation as much as possible.
          Default is True.
        """
        super().__init__()
        self._n_components = n_components
        self._demean = demean
        self._rolling_timeframe = rolling_timeframe
        self._speedup = speedup
        self._model = PCA(n_components=n_components, demean=demean, speedup=speedup)

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
    ) -> object:
        """
        Fit the returns into the risk model.

        Parameters
        ----------
        X: pandas.DataFrame or numpy.ndarray
          Instrument returns where the rows are the instruments
          and the columns are the date / time in ascending order.
          For example, if there are N instruments and T days of
          returns, the input is with the dimension of (N, T).

        Returns
        -------
        object
          The object itself.
        """
        self._factor_exposures = {}
        self._factors = {}
        self._residual_returns = {}

        for index in range(0, X.shape[1]):
            start_index = index
            end_index = index + self._rolling_timeframe + 1
            if end_index > X.shape[1]:
                break

            if isinstance(X, pd.DataFrame):
                X_input = X.iloc[:, start_index:end_index]
                index_name = X.columns[end_index - 1]
            elif isinstance(X, np.ndarray):
                X_input = X[:, start_index:end_index]
                index_name = end_index - 1

            result = self._model.fit(X_input)
            self._factor_exposures[index_name] = result.factor_exposures
            self._factors[index_name] = result.factors
            self._residual_returns[index_name] = result.residual_returns

        return self
