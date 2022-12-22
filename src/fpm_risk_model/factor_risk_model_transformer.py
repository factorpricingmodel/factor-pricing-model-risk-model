from numpy import ndarray
from pandas import DataFrame

from .factor_risk_model import FactorRiskModel
from .regressor import WLS


class FactorRiskModelTransformer:
    """
    Factor risk model transformer.
    """

    def __init__(self, regressor=None, **kwargs):
        """
        Constructor.
        """
        self._regressor = (regressor or WLS)(**kwargs)

    def transform(self, risk_model: object, y: ndarray) -> object:
        """
        Transform
        """
        factor_returns = risk_model.factor_returns
        if not isinstance(factor_returns, (ndarray, DataFrame)):
            raise TypeError(
                "Factor returns should be in numpy ndarray type, but got "
                f"{factor_returns.__class__.__name__}. If it is a rolling "
                "risk model, please use `RollingFactorRiskModelTransformer` "
                "instead"
            )

        X = factor_returns
        B = risk_model.factor_exposures
        y_input = y
        if isinstance(factor_returns, DataFrame):
            X = X.values
            B = B.values
            y_input = y.values

        factor_exposures = self._regressor.fit(X=X, y=y_input)
        residual_returns = y_input - X @ B

        if isinstance(factor_returns, DataFrame):
            factor_exposures = DataFrame(
                factor_exposures,
                index=risk_model.factor_exposures.index,
                columns=y.columns,
            )
            residual_returns = DataFrame(
                residual_returns,
                index=y.index,
                columns=y.columns,
            )

        return FactorRiskModel(
            factor_exposures=factor_exposures,
            factor_returns=factor_returns,
            factor_covariances=risk_model.factor_covariances,
            residual_returns=residual_returns,
        )


class RollingFactorRiskModelTransformer:
    """
    Rolling factor risk model transformer.
    """

    def __init__(self, rolling_timeframe: int, regressor=None, **kwargs):
        """
        Constructor.
        """
        self._rolling_timeframe = rolling_timeframe
        self._transformer = FactorRiskModelTransformer(regressor=regressor, **kwargs)

    def transform(self, risk_model: object, y: ndarray) -> object:
        """
        Transform
        """
        factor_returns = risk_model.factor_returns
        if not isinstance(factor_returns, dict):
            raise TypeError(
                "Factor returns should be in dict type, but got "
                f"{factor_returns.__class__.__name__}. "
            )

        T = y.shape[0]
        factor_exposures = {}
        residual_returns = {}
        for index in range(0, T):
            start_index = index
            end_index = index + self._rolling_timeframe + 1
            if end_index > T:
                break

            if isinstance(y, DataFrame):
                y_input = y.iloc[start_index:end_index, :]
                index_name = y.index[end_index - 1]
            elif isinstance(y, ndarray):
                y_input = y[start_index:end_index, :]
                index_name = end_index - 1

            if index_name not in factor_returns:
                raise ValueError(
                    f"Index {index_name} cannot be found in the given "
                    "risk model. The risk model cannot be transformed "
                    "by the given returns"
                )

            index_factor_returns = factor_returns[index_name]
            index_factor_exposures = self._regressor.fit(
                X=index_factor_returns, y=y_input
            )
            index_residual_returns = (
                y_input - index_factor_exposures.T @ index_factor_returns
            )

            factor_exposures[index_name] = index_factor_exposures
            residual_returns[index_name] = index_residual_returns

        return FactorRiskModel(
            factor_exposures=factor_exposures,
            factors_returns=factor_returns,
            factor_covariances=risk_model.factor_covariances,
            residual_returns=residual_returns,
        )
