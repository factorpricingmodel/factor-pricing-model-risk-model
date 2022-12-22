from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA as sklearn_PCA

from ..factor_risk_model import FactorRiskModel
from ..regressor import WLS


class PCA(FactorRiskModel):
    def __init__(
        self,
        n_components: int,
        demean: Optional[bool] = True,
        speedup: Optional[bool] = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_components : int
          Number of components.
        demean : Optional[bool]
          Indicate whether to demean before fitting. Default is True.
        speedup: Optional[bool]
          Indicate whether to speed up the computation as much as possible.
          Default is True.
        """
        super().__init__()
        self._n_components = n_components
        self._demean = demean
        self._speedup = speedup
        self._model = sklearn_PCA(n_components=n_components)

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> object:
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
        # First convert all the numpy ndarray type first
        if isinstance(X, pd.DataFrame):
            X_fit = X.values
        elif isinstance(X, np.ndarray):
            X_fit = X
        else:
            raise TypeError(
                "X must be in numpy ndarray or pandas DataFrame type, "
                f"not {X.__class__.__name__}"
            )

        # Normalize the instrument return by full history mean
        if self._demean:
            X_mean = np.array(np.mean(X, axis=0))[np.newaxis, :]
            X_fit = np.subtract(X_fit, X_mean)

        # Remove the instruments without any returns always
        if self._speedup:
            # Select the instruments of which the returns are not always 0
            X_reindex = ~np.all(np.abs(X_fit) < 1e-20, axis=0)
            X_fit = X_fit[:, X_reindex]

        # Fit with skilearn PCA on the return matrix (T, N)
        self._model.fit(X_fit)
        # N is the number of instruments and T is the number of time frames
        T = X.shape[0]
        N = X.shape[1]
        # Dimension (n, N) where n is the number of instruments
        # Eigenvectors
        U_m = self._model.components_
        # Exposure matrix (n, N)
        B = np.multiply(
            U_m,
            np.array(self._model.singular_values_ * np.sqrt(T))[:, np.newaxis],
        )
        # Factor matrix (T, n)
        wls = WLS()
        F = wls.fit(X=B.T, y=X_fit.T).T
        # Residual returns (N, T)
        residual_returns = X_fit - F @ B
        # Factor covariance matrix
        F_cov = np.cov(F.T)

        # Fill back the instruments which don't have any returns
        # with 0.0 exposures and residual returns
        if self._speedup:
            B_reindex = np.zeros((self._n_components, N))
            residual_returns_reindex = np.zeros(X.shape)
            B_reindex[:, X_reindex] = B[:, :]
            residual_returns_reindex[:, X_reindex] = residual_returns[:, :]
            B = B_reindex
            residual_returns = residual_returns_reindex

        # Convert back to dataframe if necessary
        if isinstance(X, pd.DataFrame):
            factor_index = [
                f"factor_{index + 1}" for index in range(self._n_components)
            ]
            B = pd.DataFrame(B, index=factor_index, columns=X.columns)
            F = pd.DataFrame(F, index=X.index, columns=factor_index)
            residual_returns = pd.DataFrame(
                residual_returns, index=X.index, columns=X.columns
            )
            F_cov = pd.DataFrame(F_cov, index=factor_index, columns=factor_index)

        # Return itself out
        self._factor_exposures = B
        self._factor_returns = F
        self._residual_returns = residual_returns
        self._factor_covariances = F_cov
        return self
