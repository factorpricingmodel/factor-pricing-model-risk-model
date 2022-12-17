import logging
from typing import Optional, Union

import numpy as np
import pandas as pd

from ..factor_risk_model import FactorRiskModel
from .pca import PCA

LOGGER = logging.getLogger(__name__)


class RollingPCA(FactorRiskModel):
    def __init__(
        self,
        n_components: int,
        rolling_timeframe: int,
        demean: Optional[bool] = True,
        speedup: Optional[bool] = True,
        show_progress: Optional[bool] = False,
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
        show_progress: Optional[bool]
          Show the progress in interactive mode.
        """
        super().__init__()
        self._n_components = n_components
        self._demean = demean
        self._rolling_timeframe = rolling_timeframe
        self._speedup = speedup
        self._show_progress = show_progress
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
          Instrument returns where the columns are the instruments
          and the index is the date / time in ascending order.
          For example, if there are N instruments and T days of
          returns, the input is with the dimension of (T, N).

        Returns
        -------
        object
          The object itself.
        """
        self._factor_exposures = None
        self._factors = None
        self._residual_returns = None
        factor_exposures = {}
        factor_returns = {}
        residual_returns = {}

        T = X.shape[0]
        iterator = range(0, T)
        if self._show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator)

        try:
            for index in iterator:
                start_index = index
                end_index = index + self._rolling_timeframe + 1
                if end_index > T:
                    break

                if isinstance(X, pd.DataFrame):
                    X_input = X.iloc[start_index:end_index, :]
                    index_name = X.index[end_index - 1]
                elif isinstance(X, np.ndarray):
                    X_input = X[start_index:end_index, :]
                    index_name = end_index - 1

                result = self._model.fit(X_input)
                factor_exposures[index_name] = result.factor_exposures
                factor_returns[index_name] = result.factor_returns
                residual_returns[index_name] = result.residual_returns
        except Exception:
            LOGGER.exception(
                f"Failed to fit on the index {index}. For details, please refer to "
                "the following error message."
            )
            raise

        self._factor_exposures = factor_exposures
        self._factor_returns = factor_returns
        self._residual_returns = residual_returns
        return self
