from typing import Union, Optional

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA as sklearn_PCA


class PCA:
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
        self._n_components = n_components
        self._demean = demean
        self._speedup = speedup
        self._model = sklearn_PCA(n_components=n_components)
        self._factor_exposures = None
        self._factors = None
        self._residual_returns = None

    @property
    def factor_exposures(self) -> np.ndarray:
        """
        Return the factor exposures.

        Return
        ------
        np.ndarray
          Matrix in dimension (N, n) where N is the number of
          instruments and n is the number of components in PCA.
        """
        return self._factor_exposures

    @property
    def factors(self) -> np.ndarray:
        """
        Return the factors.

        Return
        ------
        np.ndarray
          Matrix in dimension (n, T) where n is the number of
          components in PCA and T is the number of time frames.
        """
        return self._factors

    @property
    def residual_returns(self) -> np.ndarray:
        """
        Return the residual returns.

        Return
        ------
        np.ndarray
          Matrix in dimension (N, T) where N is the number of
          instruments and T is the number of time frames.
        """
        return self._residual_returns

    def fit(self, X: Union[np.ndarray, pd.DataFrame]) -> object:
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
            X_mean = np.array(np.mean(X, axis=1))[:, np.newaxis]
            X_fit = np.subtract(X_fit, X_mean)

        # Remove the instruments without any returns always
        if self._speedup:
            # Select the instruments of which the returns are not always 0
            X_reindex = ~np.all(np.abs(X_fit) < 1e-20, axis=1)
            X_fit = X_fit[X_reindex, :]

        # Fit on the covariance matrix with dimension (N, N)
        self._model.fit(X_fit.T)
        # Dimension (N, n) where n is the number of instruments
        # Eigenvectors
        U_m = self._model.components_.T
        # Exposure matrix (N, n)
        B = np.multiply(
            U_m,
            np.array(self._model.singular_values_ * np.sqrt(X.shape[1]))[np.newaxis, :],
        )
        # Factor matrix (n, T)
        F = np.linalg.pinv(B.T @ B) @ B.T @ X_fit
        # Residual returns (N, T)
        residual_returns = X_fit - B @ F

        # Fill back the instruments which don't have any returns
        # with 0.0 exposures and residual returns
        if self._speedup:
            B_reindex = np.zeros((X.shape[0], self._n_components))
            residual_returns_reindex = np.zeros(X.shape)
            B_reindex[X_reindex, :] = B[:, :]
            residual_returns_reindex[X_reindex, :] = residual_returns[:, :]
            B = B_reindex
            residual_returns = residual_returns_reindex

        # Convert back to dataframe if necessary
        if isinstance(X, pd.DataFrame):
            B = pd.DataFrame(B, index=X.index)
            F = pd.DataFrame(F, columns=X.columns)
            residual_returns = pd.DataFrame(
                residual_returns, index=X.index, columns=X.columns
            )

        # Return itself out
        self._factor_exposures = B
        self._factors = F
        self._residual_returns = residual_returns
        return self


class RollingPCA:
    def __init__(
        self,
        n_components: int,
        demean: bool,
        rolling_timeframe: int,
        fillna_zero: Optional[bool] = True,
    ):
        """
        Constructor.

        Parameters
        ----------
        n_components : int
          Number of components.
        demean : bool
          Indicate whether to demean before fitting.
        rolling_timeframe: int
          Number of rolling time frames.
        fillna_zero: bool
          Fill the nan to 0.0 always. Default is True.
        """
        self._n_components = n_components
        self._demean = demean
        self._rolling_timeframe = rolling_timeframe
        self._fillna_zero = fillna_zero
        self._model = PCA(n_components=n_components, demean=demean)
        self._factor_exposures = {}
        self._factors = {}
        self._residual_returns = {}

    def fit(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        validity: Optional[Union[np.ndarray, pd.DataFrame]] = None,
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
        if validity is not None and X.shape != validity.shape:
            raise ValueError(
                f"Dimension of X {X.shape} is different than "
                f"dimension of validity {validity.shape}"
            )
        """
        self._factor_exposures = {}
        self._factors = {}
        self._residual_returns = {}

        for index in range(0, X.shape[1]):
            start_index = index
            end_index = index + self._rolling_timeframe + 1
            if end_index >= X.shape[1]:
                break

            X_input = X[:, start_index:end_index]
            if validity:
                X_input = X_input.where(validity[start_index:end_index])

            if self._fillna_zero:
                X_input = np.nan_to_num(X_input)

            index_name = index
            if isinstance(X, pd.DataFrame):
                index_name = X.columns[end_index - 1]

            result = self._model.fit(X)
            self._factor_exposures[index_name] = result.factor_exposures
            self._factors[index_name] = result.factors
            self._residual_returns[index_name] = result.residual_returns

        return self
