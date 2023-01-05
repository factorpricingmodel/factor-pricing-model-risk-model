from numpy import array, nan
from pandas import Series
from pandas.testing import assert_series_equal

from fpm_risk_model.accuracy.value_at_risk import (
    compute_value_at_risk_breach_statistics,
    compute_value_at_risk_rolling_breach_statistics,
    compute_value_at_risk_threshold,
)


def test_compute_value_at_risk_threshold(weights, rolling_factor_risk_model):
    var_threshold = compute_value_at_risk_threshold(
        weights=weights,
        rolling_risk_model=rolling_factor_risk_model,
        threshold=0.95,
    )
    expected_var_threshold = Series(
        array(
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                0.0,
                0.02711031,
                0.0337584,
                0.03866137,
                0.03303078,
            ]
        ),
        index=weights.index,
    )
    assert_series_equal(var_threshold, expected_var_threshold)


def test_compute_value_at_risk_breach_statistics(
    daily_returns, weights, rolling_factor_risk_model
):
    var_breach_statistics = compute_value_at_risk_breach_statistics(
        X=daily_returns,
        weights=weights,
        rolling_risk_model=rolling_factor_risk_model,
        threshold=0.95,
    )
    expected_var_breach_statistics = Series(
        [False, False, False, False, False, True, False, True, False, False],
        index=weights.index,
        dtype=bool,
    )
    assert_series_equal(var_breach_statistics, expected_var_breach_statistics)


def test_compute_value_at_risk_rolling_breach_statistics(
    daily_returns, weights, rolling_factor_risk_model
):
    var_rolling_breach_statistics = compute_value_at_risk_rolling_breach_statistics(
        X=daily_returns,
        weights=weights,
        rolling_risk_model=rolling_factor_risk_model,
        threshold=0.95,
        rolling_timeframe=3,
        min_periods=0,
    )
    expected_var_rolling_breach_statistics = Series(
        array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.33333333,
                0.33333333,
                0.66666667,
                0.33333333,
                0.33333333,
            ]
        ),
        index=weights.index,
    )
    assert_series_equal(
        var_rolling_breach_statistics, expected_var_rolling_breach_statistics
    )
