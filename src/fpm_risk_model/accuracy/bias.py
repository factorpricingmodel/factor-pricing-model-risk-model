from typing import Optional

from numpy import ndarray, sqrt, sum
from pandas import Series

from ..rolling_factor_risk_model import RollingFactorRiskModel


def compute_standardized_returns(
    X: ndarray,
    weights: ndarray,
    rolling_risk_model: RollingFactorRiskModel,
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

    Returns
    -------
    Series
        A timeseries of standardized returns.
    """
    b_t = Series()
    for index, risk_model in rolling_risk_model.items():
        index_weights = weights.loc[index, :]
        returns = sum(X.loc[index, :] * index_weights)
        vol = sqrt((risk_model.cov().fillna(0.0) @ index_weights) @ index_weights)
        b_t[index] = returns / vol

    return b_t


def compute_bias_statistics(
    X: ndarray,
    weights: ndarray,
    rolling_risk_model: RollingFactorRiskModel,
    rolling_timeframe: int,
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
    )
    return standardized_returns.rolling(
        rolling_timeframe, min_periods=min_periods
    ).std()
