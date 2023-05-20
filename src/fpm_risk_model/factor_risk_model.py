import json
from os.path import join
from typing import Optional

from numpy import any, diag_indices_from, nan, ndarray
from pandas import DataFrame, Series

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
        residual_returns : ndarray
          Residual returns of the factor risk model.
        """
        super().__init__(**kwargs)
        self._factor_exposures = factor_exposures
        self._factor_returns = factor_returns
        self._residual_returns = residual_returns

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

    def fit(self, X: ndarray, weights: Optional[ndarray] = None) -> object:
        """
        Fit the model.

        Parameters
        ----------
        X : ndarray
          Input array of shape (T, N) where N is the number of
          instruments and T is the number of timeframes.

        weights: Optional[ndarray]
          Weights array of shape (N,) where N is the number of
          instruments.
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
            factor_returns=self._factor_returns.copy(),
            residual_returns=self._residual_returns.copy(),
            **self._config.dict(),
        )

    def specific_variances(self, weights=None, ddof=1) -> ndarray:
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
        eg = self._engine
        T = self._residual_returns.shape[0]
        residual_returns = self._residual_returns
        if isinstance(self._residual_returns, DataFrame):
            residual_returns = residual_returns.values

        if weights is not None:
            r_mean = eg.mean(residual_returns * weights[:, eg.newaxis], axis=0)
            variances = (residual_returns - r_mean) ** 2
            variances *= weights[:, eg.newaxis]
        else:
            r_mean = eg.mean(residual_returns, axis=0)
            variances = (residual_returns - r_mean) ** 2

        variances = sum(variances) / (T - ddof)

        if isinstance(self._residual_returns, DataFrame):
            variances = Series(variances, index=self._residual_returns.columns)

        return variances

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
        regressor_result = regressor.fit(X=X, y=y_input)
        factor_exposures = regressor_result.beta
        residual_returns = regressor_result.alpha

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

    def cov(self, halflife: Optional[float] = None, ddof=1) -> ndarray:
        """
        Get the covariance matrix.

        Parameters
        ----------
        halflife : Optional[float]
            Half life in applying the exponential weighting on factor
            returns for computing the factor covariance matrix. If
            None is passed, no exponential weighting is applied.

        Returns
        -------
        numpy.ndarray
            A square pairwise covariance matrix which its
            diagonal entries are the variances.
        """
        eg = self._engine
        B = self._factor_exposures
        F = self._factor_returns
        if F is None:
            raise ValueError("Factor return cannot be None")
        elif isinstance(F, DataFrame):
            F = F.values

        W = None
        T = F.shape[0]
        if halflife is not None:
            W = eg.array([2 ** (-(T - 1 - t) / halflife) for t in range(0, T)])
            F = F * (W[:, eg.newaxis] ** 0.5)

        F = F - eg.mean(F, axis=0)
        factor_covariances = (F.T @ F) / (T - ddof)
        specific_variances = self.specific_variances(weights=W, ddof=ddof)

        R = specific_variances
        if isinstance(B, DataFrame):
            instruments = self._factor_exposures.columns
            B = B.values
            R = R.loc[instruments].values

        if not isinstance(B, ndarray):
            raise TypeError(
                "Only pandas DataFrame / numpy ndarray is supported, but not "
                f"{B.__class__.__name__}"
            )

        cov = B.T @ factor_covariances @ B

        # Add the specific variances into the covariance matrix
        cov[diag_indices_from(cov)] += R

        # Set zero covariance instruments to nan
        valid_instruments = any(cov != 0.0, axis=0)
        cov[~valid_instruments, :] = nan
        cov[:, ~valid_instruments] = nan

        if not self._config.show_all_instruments:
            cov = cov[valid_instruments, :][:, valid_instruments]
            if isinstance(self._factor_exposures, DataFrame):
                instruments = instruments[valid_instruments]

        if isinstance(self._factor_exposures, DataFrame):
            cov = DataFrame(cov, index=instruments, columns=instruments)

        return cov

    def write_directory(self, path: str, format="parquet", **kwargs):
        """
        Write the factor risk model to directory.

        Parameters
        ----------
        path: str
          Destination path.

        format: str
            Supported formats. Default is "parquet". Options
            are "csv", "parquet" and "hdf".

        **kwargs: dict
            Optional keyword arguments for the write operation.
        """
        method_name = f"to_{format}"
        getattr(self._factor_exposures, method_name)(
            join(path, f"factor_exposures.{format}"), **kwargs
        )
        getattr(self._factor_returns, method_name)(
            join(path, f"factor_returns.{format}"), **kwargs
        )
        getattr(self._residual_returns, method_name)(
            join(path, f"residual_returns.{format}"), **kwargs
        )
        with open(join(path, "metadata.json"), mode="w+") as fp:
            json.dump(self.asdict(), fp)

    @classmethod
    def read_directory(cls, path: str, format: str = "parquet", **kwargs):
        """
        Read model from specified directory.

        Parameters
        ----------
        path: str
            Directory to read model from.

        format: str
            Supported formats. Default is "parquet". Options
            are "csv", "parquet" and "hdf".

        **kwargs: dict
            Optional keyword arguments for the write operation.
        """
        method_name = f"read_{format}"
        import pandas

        method = getattr(pandas, method_name)
        factor_exposures = method(join(path, f"factor_exposures.{format}"), **kwargs)
        factor_exposures.index.name = None
        factor_returns = method(join(path, f"factor_returns.{format}"), **kwargs)
        factor_returns.index.name = None
        residual_returns = method(join(path, f"residual_returns.{format}"), **kwargs)
        residual_returns.index.name = None
        with open(join(path, "metadata.json")) as fp:
            metadata = json.load(fp)

        return cls(
            factor_exposures=factor_exposures,
            factor_returns=factor_returns,
            residual_returns=residual_returns,
            **metadata,
        )
