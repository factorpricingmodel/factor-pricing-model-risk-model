from typing import Optional

from numpy import nan, ndarray, sqrt, sum
from pandas import DataFrame, Series
from scipy.stats import norm

from ..rolling_factor_risk_model import RollingFactorRiskModel


def compute_value_at_risk_threshold(
    weights: DataFrame,
    rolling_risk_model: Optional[RollingFactorRiskModel] = None,
    forecast_vols: Optional[Series] = None,
    threshold: Optional[float] = 0.95,
) -> Series:
    """
    Compute the VaR threshold for the given weights.

    Parameters
    ----------
    weights : DataFrame
        The portfolio weights for each instrument. The input
        index and columns are the date / time and instruments
        respectively. The weights should be normalized, i.e.
        sum to one for each time frame.

    rolling_risk_model: Optional[RollingFactorRiskModel]
        The rolling risk model.

    forecast_vols: Optional[Series]
        The forecast volatility of instruments.

    threshold: float
        The threshold for the VaR. The value should be between 0
        and 1. Default is 95%.
    """
    if not (0.0 < threshold < 1.0):
        raise ValueError(f"Threshold {threshold} should be between 0 and 1")
    quantile = norm.ppf(threshold)
    value_at_risk = Series(nan, index=weights.index)
    instruments = weights.columns
    for index, index_weights in weights.iterrows():
        if rolling_risk_model is not None:
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
        else:
            vol = forecast_vols[index]
        value_at_risk[index] = quantile * vol

    return value_at_risk


def compute_value_at_risk_breach_statistics(
    X: DataFrame,
    weights: DataFrame,
    rolling_risk_model: Optional[RollingFactorRiskModel] = None,
    forecast_vols: Optional[Series] = None,
    threshold: Optional[float] = 0.95,
) -> Series:
    """
    Compute the VaR breach statistics.

    Parameters
    ----------
    X: DataFrame
        Instrument returns. The input index and columns are
        date / time and instruments respectively.

    weights : DataFrame
        The portfolio weights for each instrument. The input
        index and columns are the date / time and instruments
        respectively. The weights should be normalized, i.e.
        sum to one for each time frame.

    rolling_risk_model: Optional[RollingFactorRiskModel]
        The rolling risk model.

    forecast_vols: Optional[Series]
        The forecast volatility of instruments.

    threshold: float
        The threshold for the VaR. The value should be between 0
        and 1. Default is 95%.
    """
    value_at_risk_threshold = compute_value_at_risk_threshold(
        weights=weights,
        rolling_risk_model=rolling_risk_model,
        forecast_vols=forecast_vols,
        threshold=threshold,
    )
    portfolio_returns = sum(X * weights, axis=1)[value_at_risk_threshold.index]
    return portfolio_returns.le(-value_at_risk_threshold)


def compute_value_at_risk_rolling_breach_statistics(
    X: ndarray,
    weights: ndarray,
    rolling_timeframes: int,
    rolling_risk_model: Optional[RollingFactorRiskModel] = None,
    forecast_vols: Optional[Series] = None,
    threshold: Optional[float] = 0.95,
    min_periods: Optional[int] = None,
) -> Series:
    """
    Compute the VaR breach statistics.

    Parameters
    ----------
    X: DataFrame
        Instrument returns. The input index and columns are
        date / time and instruments respectively.

    weights : DataFrame
        The portfolio weights for each instrument. The input
        index and columns are the date / time and instruments
        respectively. The weights should be normalized, i.e.
        sum to one for each time frame.

    rolling_timeframes: int
        The number of rolling time frames to compute the percentage
        of returns breaching the specified VaR.

    rolling_risk_model: Optional[RollingFactorRiskModel]
        The rolling risk model.

    forecast_vols: Optional[Series]
        The forecast volatility of instruments.

    threshold: float
        The threshold for the VaR. The value should be between 0
        and 1. Default is 95%.
    """
    breach_statistics = compute_value_at_risk_breach_statistics(
        X=X,
        weights=weights,
        rolling_risk_model=rolling_risk_model,
        forecast_vols=forecast_vols,
        threshold=threshold,
    )
    return (
        breach_statistics.rolling(rolling_timeframes, min_periods=min_periods).sum()
        / rolling_timeframes
    )
