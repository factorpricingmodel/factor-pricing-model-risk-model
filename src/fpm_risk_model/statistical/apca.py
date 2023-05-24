from typing import Optional, Union

from numpy import ndarray
from pandas import DataFrame
from sklearn.decomposition import PCA as sklearn_PCA

from ..factor_risk_model import FactorRiskModel
from ..regressor import WLS


class APCAConfig(FactorRiskModel.ConfigClass):
    """
    Asymptotic PCA statistics model configuration class.


    Parameters
    ----------
    n_components : int
        Number of components.
    demean : Optional[bool]
        Indicate whether to demean before fitting. Default is True.
    """

    n_components: Union[int, float, str]
    demean: Optional[bool] = True


class APCA(FactorRiskModel):
    """
    Asymptotic PCA statistics model.
    """

    ConfigClass = APCAConfig

    def __init__(
        self,
        n_components: int,
        demean: Optional[bool] = True,
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
        """
        super().__init__(n_components=n_components, demean=demean, **kwargs)
        self._model = sklearn_PCA(n_components=n_components)

    def fit(
        self,
        X: Union[ndarray, DataFrame],
        atol: float = 1e-10,
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
        N = X.shape[1]

        # Initialize the engine
        eg = self._engine

        # Normalize the instrument return by full history mean
        if self._config.demean:
            X_mean = eg.array(eg.mean(X, axis=0))[eg.newaxis, :]
            X_fit = eg.subtract(X_fit, X_mean)

        # Remove the instruments without any returns always
        # Select the instruments of which the returns are not always 0
        X_reindex = ~eg.all(eg.abs(X_fit) < 1e-20, axis=0)
        X_fit = X_fit[:, X_reindex]

        # Factor model - R = B @ F + residual_returns
        # Fit with skilearn PCA on the return matrix (T, N) in t-space
        self._model.fit(X_fit.T)
        # Eigenvectors
        U_m = self._model.components_
        # Just choose F = U_m ^T (Shape = (T, n))
        F = U_m.T

        # B^T = (F @ F^T)^{-1} @ F @ R^T
        # B = U_m @ R (Shape = (n, N))
        # B = F.T @ scaled_X_fit
        wls = WLS()
        wls_result = wls.fit(X=F, y=X_fit)
        B = wls_result.beta
        residual_returns = wls_result.alpha

        # Fill back the instruments which don't have any returns
        # with 0.0 exposures and residual returns
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

        self._factor_exposures = B
        self._factor_returns = F
        self._residual_returns = residual_returns
        return self
