import pytest
from pandas import DataFrame, bdate_range
from pandas.testing import assert_frame_equal

from fpm_risk_model.pipeline.portfolio import equal_weighted_portfolio


@pytest.fixture(scope="module")
def instruments():
    return ["A", "AAL", "AAP", "AAPL"]


@pytest.fixture(scope="module")
def dates():
    return bdate_range("2016-01-04", "2016-01-15")


@pytest.fixture(scope="module")
def validity(dates, instruments):
    return DataFrame(
        [
            [True, True, False, True],
            [True, True, False, True],
            [True, True, False, True],
            [True, True, False, True],
            [True, True, False, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, False, False, True],
            [True, False, False, True],
            [False, False, False, False],
        ],
        index=dates,
        columns=instruments,
    )


def test_equal_weighted_portfolio(validity):
    weights = equal_weighted_portfolio(validity)
    expected_weights = DataFrame(
        [
            [1 / 3, 1 / 3, 0.0, 1 / 3],
            [1 / 3, 1 / 3, 0.0, 1 / 3],
            [1 / 3, 1 / 3, 0.0, 1 / 3],
            [1 / 3, 1 / 3, 0.0, 1 / 3],
            [1 / 3, 1 / 3, 0.0, 1 / 3],
            [1 / 2, 0.0, 0.0, 1 / 2],
            [1 / 2, 0.0, 0.0, 1 / 2],
            [1 / 2, 0.0, 0.0, 1 / 2],
            [1 / 2, 0.0, 0.0, 1 / 2],
            [0.0, 0.0, 0.0, 0.0],
        ],
        index=validity.index,
        columns=validity.columns,
    )
    assert_frame_equal(weights, expected_weights)
