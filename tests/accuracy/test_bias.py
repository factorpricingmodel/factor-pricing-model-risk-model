from numpy import array, nan
from pandas import Series
from pandas.testing import assert_series_equal

from fpm_risk_model.accuracy.bias import (
    compute_bias_statistics,
    compute_standardized_returns,
)


def test_compute_standardized_returns(
    daily_returns, weights, rolling_factor_risk_model
):
    standardized_returns = compute_standardized_returns(
        X=daily_returns,
        weights=weights,
        rolling_risk_model=rolling_factor_risk_model,
    )
    expected_standardized_returns = Series(
        array(
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                0.87928191,
                -1.7095046,
                0.80100368,
                -1.0760382,
            ]
        ),
        index=weights.index,
    )
    assert_series_equal(standardized_returns, expected_standardized_returns)


def test_compute_bias_statistics(daily_returns, weights, rolling_factor_risk_model):
    bias_statistics = compute_bias_statistics(
        X=daily_returns,
        weights=weights,
        rolling_risk_model=rolling_factor_risk_model,
        rolling_timeframe=5,
        min_periods=0,
    )
    expected_bias_statistics = Series(
        array([nan, nan, nan, nan, nan, nan, nan, 1.8305485, 1.47255984, 1.31524515]),
        index=weights.index,
    )
    assert_series_equal(expected_bias_statistics, bias_statistics)
