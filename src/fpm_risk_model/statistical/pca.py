from typing import Optional, Union

from numpy import ndarray
from pandas import DataFrame, Series
from sklearn.decomposition import PCA as sklearn_PCA

from ..factor_risk_model import FactorRiskModel
from ..regressor import WLS


class PCAConfig(FactorRiskModel.ConfigClass):
    """
    PCA statistics model configuration class.


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

    n_components: Union[int, float, str]
    demean: Optional[bool] = True
    speedup: Optional[bool] = True


class PCA(FactorRiskModel):
    """
    PCA statistics model.
    """

    ConfigClass = PCAConfig

    def __init__(
        self,
        n_components: int,
        demean: Optional[bool] = True,
        speedup: Optional[bool] = True,
        **kwargs,
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
        super().__init__(
            n_components=n_components, demean=demean, speedup=speedup, **kwargs
        )
        self._model = sklearn_PCA(n_components=n_components)

    def fit(
        self,
        X: Union[ndarray, DataFrame],
        weights: Optional[Union[ndarray, Series]] = None,
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
        # First convert all the numpy ndarray type first
        X_fit = self._to_numpy(X)
        weights_fit = self._to_numpy(weights)

        # Initialize the engine
        eg = self._engine

        # Normalize the instrument return by full history mean
        if self._config.demean:
            X_mean = eg.array(eg.mean(X, axis=0))[eg.newaxis, :]
            X_fit = eg.subtract(X_fit, X_mean)

        # Remove the instruments without any returns always
        if self._config.speedup:
            # Select the instruments of which the returns are not always 0
            X_reindex = ~eg.all(eg.abs(X_fit) < 1e-20, axis=0)
            X_fit = X_fit[:, X_reindex]
            if weights_fit is not None:
                weights_fit = weights_fit[X_reindex]

        # Fit with skilearn PCA on the return matrix (T, N)
        self._model.fit(X_fit)
        # N is the number of instruments and T is the number of time frames
        T = X.shape[0]
        N = X.shape[1]
        # Dimension (n, N) where n is the number of instruments
        # Eigenvectors
        U_m = self._model.components_
        # Exposure matrix (n, N)
        B = eg.multiply(
            U_m,
            eg.array(self._model.singular_values_ * eg.sqrt(T))[:, eg.newaxis],
        )
        # Factor matrix (T, n)
        wls = WLS()
        wls_result = wls.fit(X=B.T, y=X_fit.T, weights=weights_fit)
        F = wls_result.beta.T
        residual_returns = wls_result.alpha.T

        # Fill back the instruments which don't have any returns
        # with 0.0 exposures and residual returns
        if self._config.speedup:
            B_reindex = eg.zeros((B.shape[0], N))
            residual_returns_reindex = eg.zeros(X.shape)
            B_reindex[:, X_reindex] = B[:, :]
            residual_returns_reindex[:, X_reindex] = residual_returns[:, :]
            B = B_reindex
            residual_returns = residual_returns_reindex

        # Convert back to dataframe if necessary
        if isinstance(X, DataFrame):
            factor_index = [f"factor_{index + 1}" for index in range(B.shape[0])]
            B = DataFrame(B, index=factor_index, columns=X.columns)
            F = DataFrame(F, index=X.index, columns=factor_index)
            residual_returns = DataFrame(
                residual_returns, index=X.index, columns=X.columns
            )

        # Return itself out
        self._factor_exposures = B
        self._factor_returns = F
        self._residual_returns = residual_returns
        return self
