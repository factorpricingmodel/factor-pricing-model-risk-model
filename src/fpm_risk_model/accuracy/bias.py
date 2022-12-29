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
