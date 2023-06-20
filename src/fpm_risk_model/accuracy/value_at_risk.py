from typing import Any, Dict, Optional, Union

from numpy import nan, ndarray, sqrt, sum
from pandas import DataFrame, Series
from scipy.stats import norm

from ..rolling_factor_risk_model import RollingFactorRiskModel


def compute_value_at_risk_threshold(
    weights: DataFrame,
    rolling_risk_model: Optional[Union[RollingFactorRiskModel, Dict[Any, Any]]] = None,
    forecast_vols: Optional[Series] = None,
    threshold: Optional[float] = 0.95,
    cov_halflife: Optional[float] = None,
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

    rolling_risk_model: Union[RollingFactorRiskModel, Dict[Any, Any]]
        A rolling risk model object or dictionary of covariances of
        which the keys and values are dates and covariances.

    forecast_vols: Optional[Series]
        The forecast volatility of instruments.

    threshold: float
        The threshold for the VaR. The value should be between 0
        and 1. Default is 95%.

    cov_halflife: Optional[float]
        Halflife in computing covariances.
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
            elif isinstance(risk_model, DataFrame):
                cov = risk_model
            else:
                cov = risk_model.cov(halflife=cov_halflife)

            cov = cov.reindex(index=instruments, columns=instruments).fillna(0.0).values
            vol = sqrt((cov @ index_weights) @ index_weights)
        else:
            vol = forecast_vols[index]
        value_at_risk[index] = quantile * vol

    return value_at_risk


def compute_value_at_risk_breach_statistics(
    X: DataFrame,
    weights: DataFrame,
    rolling_risk_model: Optional[Union[RollingFactorRiskModel, Dict[Any, Any]]] = None,
    forecast_vols: Optional[Series] = None,
    threshold: Optional[float] = 0.95,
    cov_halflife: Optional[float] = None,
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

    rolling_risk_model: Union[RollingFactorRiskModel, Dict[Any, Any]]
        A rolling risk model object or dictionary of covariances of
        which the keys and values are dates and covariances.

    forecast_vols: Optional[Series]
        The forecast volatility of instruments.

    threshold: float
        The threshold for the VaR. The value should be between 0
        and 1. Default is 95%.

    cov_halflife: Optional[float]
        Halflife in computing covariances.
    """
    value_at_risk_threshold = compute_value_at_risk_threshold(
        weights=weights,
        rolling_risk_model=rolling_risk_model,
        forecast_vols=forecast_vols,
        threshold=threshold,
        cov_halflife=cov_halflife,
    )
    portfolio_returns = sum(X * weights, axis=1)[value_at_risk_threshold.index]
    return portfolio_returns.le(-value_at_risk_threshold)


def compute_value_at_risk_rolling_breach_statistics(
    X: ndarray,
    weights: ndarray,
    window: int,
    rolling_risk_model: Optional[Union[RollingFactorRiskModel, Dict[Any, Any]]] = None,
    forecast_vols: Optional[Series] = None,
    threshold: Optional[float] = 0.95,
    min_periods: Optional[int] = None,
    cov_halflife: Optional[float] = None,
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

    window: int
        The number of rolling time frames to compute the percentage
        of returns breaching the specified VaR.

    rolling_risk_model: Union[RollingFactorRiskModel, Dict[Any, Any]]
        A rolling risk model object or dictionary of covariances of
        which the keys and values are dates and covariances.

    forecast_vols: Optional[Series]
        The forecast volatility of instruments.

    threshold: Optional[float]
        The threshold for the VaR. The value should be between 0
        and 1. Default is 95%.

    cov_halflife: Optional[float]
        Halflife in computing covariances.
    """
    breach_statistics = compute_value_at_risk_breach_statistics(
        X=X,
        weights=weights,
        rolling_risk_model=rolling_risk_model,
        forecast_vols=forecast_vols,
        threshold=threshold,
        cov_halflife=cov_halflife,
    )
    return breach_statistics.rolling(window, min_periods=min_periods).sum() / window
