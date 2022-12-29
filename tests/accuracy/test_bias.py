from numpy import array, nan
from pandas import Series
from pandas.testing import assert_series_equal

from fpm_risk_model.accuracy.bias import compute_standardized_returns


def test_compute_standardized_returns(
    daily_returns, weights, rolling_factor_risk_model
):
    standardized_returns = compute_standardized_returns(
        X=daily_returns,
        weights=weights,
        rolling_risk_model=rolling_factor_risk_model,
    )
    expected_standardized_returns = Series(
        array([nan, 0.87928191, -1.7095046, 0.80100368, -1.0760382]),
        index=list(rolling_factor_risk_model.keys()),
    )
    assert_series_equal(standardized_returns, expected_standardized_returns)
