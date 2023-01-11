from typing import Optional

from numpy import any, diag_indices_from, nan, ndarray, var
from pandas import DataFrame

from .regressor import WLS
from .risk_model import RiskModel


class FactorRiskModel(RiskModel):
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
        factor_exposures: ndarray = None,
        factor_returns: ndarray = None,
        factor_covariances: ndarray = None,
        residual_returns: ndarray = None,
        **kwargs,
    ):
        """
        Constructor

        Parameters
        ----------
        factor_exposures : ndarray
          Factor exposures of the factor risk model.
        factors_returns : ndarray
          Factor returns of the factor risk model.
        factor_covariances : ndarray
          Factor covariances of the factor risk model.
        residual_returns : ndarray
          Residual returns of the factor risk model.
        """
        super().__init__(**kwargs)
        self._factor_exposures = factor_exposures
        self._factor_returns = factor_returns
        self._factor_covariances = factor_covariances
        self._residual_returns = residual_returns
        self._kwargs = kwargs

    @property
    def factor_exposures(self) -> ndarray:
        """
        Return the factor exposures.

        Return
        ------
        ndarray
          Matrix in dimension (n, N) where N is the number of
          instruments and n is the number of factors.
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
          factors and T is the number of time frames.
        """
        return self._factor_returns

    @property
    def factor_covariances(self) -> ndarray:
        """
        Return the factor returns.

        Return
        ------
        ndarray
          Matrix in dimension (n, n) where n is the number of
          factors.
        """
        return self._factor_covariances

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

    def fit(self, X: ndarray) -> object:
        """
        Fit the model.
        """
        pass

    def copy(self) -> object:
        """
        Copy the model.

        Returns
        -------
        object
          Copy of the model.
        """
        return FactorRiskModel(
            factor_exposures=self._factor_exposures.copy(),
            factor_covariances=self._factor_covariances.copy(),
            factor_returns=self._factor_returns.copy(),
            residual_returns=self._residual_returns.copy(),
            **self._kwargs,
        )

    def specific_variances(self, ddof=1) -> ndarray:
        """
        Get specific variances.

        Parameters
        ----------
        ddof : int
          Degrees of freedom.

        Returns
        -------
        ndarray
          Specific variances of the instruments.
        """
        if isinstance(self._residual_returns, ndarray):
            return var(self._residual_returns, axis=0, ddof=ddof)
        elif isinstance(self._residual_returns, DataFrame):
            return self._residual_returns.var(ddof=ddof)

        raise TypeError(
            "Only pandas DataFrame / numpy ndarray is supported, but not "
            f"{self._residual_returns.__class__.__name__}"
        )

    def transform(self, y: ndarray, regressor: Optional[object] = None) -> object:
        """
        Transform the factor risk model.

        The method is used to transform the factor risk model by
        passing another set of returns. Most of the time, the
        factor risk model is fitted by the estimation universe,
        and then transformed by the model universe.

        Parameters
        ----------
        y : ndarray
            The instrument returns.

        regressor : object, default=None
            Regressor to transform the input y into factor exposures.
            If None, the regressor is set to the default WLS.

        Returns
        -------
        ndarray
            The transformed factor risk model.
        """
        X = self.factor_returns
        if X is None:
            raise ValueError("Factor returns must be initialised first")
        if not isinstance(X, (ndarray, DataFrame)):
            raise TypeError(
                "Factor returns should be in numpy ndarray type, but got "
                f"{X.__class__.__name__}. If it is a rolling "
                "risk model, please use `RollingFactorRiskModelTransformer` "
                "instead"
            )

        # Convert the factor returns into a ndarray first
        if isinstance(X, DataFrame):
            X = X.values

        # Convert the y input to a ndarray first
        y_input = y
        if isinstance(y, DataFrame):
            y_input = y.values

        # Set the default regressor
        regressor = regressor or WLS()

        # Transform the factor exposures from the y input
        factor_exposures = regressor.fit(X=X, y=y_input)
        residual_returns = y_input - X @ factor_exposures

        if isinstance(self.factor_returns, DataFrame):
            factor_exposures = DataFrame(
                factor_exposures,
                index=self.factor_exposures.index,
                columns=y.columns,
            )
            residual_returns = DataFrame(
                residual_returns,
                index=y.index,
                columns=y.columns,
            )

        self._factor_exposures = factor_exposures
        self._residual_returns = residual_returns
        return self

    def cov(self) -> ndarray:
        """
        Get the covariance matrix.

        Returns
        -------
        numpy.ndarray
            A square pairwise covariance matrix which its
            diagonal entries are the variances.
        """
        cov = (
            self._factor_exposures.T @ self._factor_covariances @ self._factor_exposures
        )
        specific_variances = self.specific_variances()

        if isinstance(cov, DataFrame):
            cov_values = cov.values
            specific_variances = specific_variances.loc[cov.index]
        elif isinstance(cov, ndarray):
            cov_values = cov
        else:
            raise TypeError(
                "Only pandas DataFrame / numpy ndarray is supported, but not "
                f"{cov.__class__.__name__}"
            )

        # Add the specific variances into the covariance matrix
        cov_values[diag_indices_from(cov_values)] += specific_variances

        valid_instruments = any(cov_values != 0.0, axis=0)

        if self._show_all_instruments:
            # Set zero covariance instruments to nan
            cov_values[~valid_instruments, :] = nan
            cov_values[:, ~valid_instruments] = nan
        elif isinstance(cov, DataFrame):
            cov = cov.loc[valid_instruments, valid_instruments]
        elif isinstance(cov, ndarray):
            cov = cov[valid_instruments, :][:, valid_instruments]
        else:
            raise TypeError(
                "Only pandas DataFrame / numpy ndarray is supported, but not "
                f"{cov.__class__.__name__}"
            )

        return cov
