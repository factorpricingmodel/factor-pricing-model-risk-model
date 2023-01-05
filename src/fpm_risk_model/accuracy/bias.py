from typing import Optional

from numpy import nan, sqrt, sum
from pandas import DataFrame, Series

from ..rolling_factor_risk_model import RollingFactorRiskModel


def compute_standardized_returns(
    X: DataFrame,
    weights: DataFrame,
    rolling_risk_model: Optional[RollingFactorRiskModel] = None,
    forecast_vols: Optional[Series] = None,
) -> Series:
    """
    Compute the standardized returns given the rolling risk model.

    Standardized return is defined as

      .. math::
        b_t = \frac{r_t}{\\sigma_t}

    Parameters
    ----------
    X: ndarray
        The instrument forecast returns.
    weights: ndarray
        Weights of the instruments.
    rolling_risk_model: RollingFactorRiskModel
        The rolling risk model.
    forecast_vols: Series
        The forecast volatility.

    Returns
    -------
    Series
        A timeseries of standardized returns.
    """
    b_t = Series(nan, index=weights.index)
    instruments = weights.columns
    for index, index_weights in weights.iterrows():
        returns = sum(X.loc[index, instruments] * index_weights)
        if forecast_vols is not None:
            vol = forecast_vols.loc[index]
        else:
            risk_model = rolling_risk_model.get(index)
            if risk_model is None:
                continue
            cov = (
                risk_model.cov()
                .reindex(index=instruments, columns=instruments)
                .fillna(0.0)
                .values
            )
            vol = sqrt((cov @ index_weights) @ index_weights)
        b_t[index] = returns / vol

    return b_t


def compute_bias_statistics(
    X: DataFrame,
    weights: DataFrame,
    rolling_timeframe: int,
    rolling_risk_model: Optional[RollingFactorRiskModel] = None,
    forecast_vols: Optional[Series] = None,
    min_periods: Optional[int] = None,
) -> Series:
    """
    Compute the bias statistics.

    Standardized return is defined as

     .. math::
        b_t = \\frac{r_t}{\\sigma_t}

    and the bias statistic is expressed as

     ..math::
        B_T(t) = \\sqrt{\\frac{1}{T} \\sigma_{\\tau}(b_{\\tau} - \bar{b})^2

    Parameters
    ----------
    X: ndarray
        The instrument forecast returns.
    weights: ndarray
        Weights of the instruments.
    forecast_vols: Optional[Series]
        The forecast volatility.
    rolling_risk_model: RollingFactorRiskModel
        The rolling risk model.

    Returns
    -------
    Series
        A timeseries of bias statistic.
    """
    standardized_returns = compute_standardized_returns(
        X=X,
        weights=weights,
        rolling_risk_model=rolling_risk_model,
        forecast_vols=forecast_vols,
    )
    return standardized_returns.rolling(
        rolling_timeframe, min_periods=min_periods
    ).std()
