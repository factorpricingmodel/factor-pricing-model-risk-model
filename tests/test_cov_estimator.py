import pytest
from numpy import array
from pandas import DataFrame, Series, bdate_range
from pandas.testing import assert_frame_equal

from fpm_risk_model.cov_estimator import CovarianceEstimator
from fpm_risk_model.factor_risk_model import FactorRiskModel


@pytest.fixture(scope="module")
def instruments():
    return ["A", "AAL", "AAP", "AAPL"]


@pytest.fixture(scope="module")
def valid_instruments():
    return ["A", "AAL", "AAPL"]


@pytest.fixture(scope="module")
def dates():
    return bdate_range("2016-01-04", "2016-01-15")


@pytest.fixture(scope="module")
def factors():
    return ["factor_1", "factor_2"]


@pytest.fixture(scope="module")
def factor_exposures():
    return array(
        [
            [-0.15454215, -0.22795166, 0.0, -0.17179763],
            [0.00706732, 0.08354979, 0.0, -0.11721647],
        ]
    )


@pytest.fixture(scope="module")
def factor_returns():
    return array(
        [
            [0.06323026, -0.15644581],
            [0.01829957, 0.09617634],
            [-0.06074615, 0.17670851],
            [0.1238173, 0.14189995],
            [-0.03715707, -0.04710242],
            [-0.08798148, -0.03209175],
            [-0.1300186, 0.00469255],
            [0.14264733, -0.05445549],
            [-0.13802269, -0.07709159],
            [0.10593153, -0.05229029],
        ]
    )


@pytest.fixture(scope="module")
def residual_returns():
    return array(
        [
            [-0.00422976, 0.00199051, 0.0, 0.00116378],
            [0.01038799, -0.00488857, 0.0, -0.00285816],
            [0.00548287, -0.00258023, 0.0, -0.00150856],
            [-0.01266259, 0.00595899, 0.0, 0.003484],
            [-0.00424175, 0.00199616, 0.0, 0.00116708],
            [-0.01853336, 0.00872176, 0.0, 0.00509929],
            [-0.00185692, 0.00087386, 0.0, 0.00051091],
            [-0.00071556, 0.00033674, 0.0, 0.00019688],
            [0.01124235, -0.00529063, 0.0, -0.00309323],
            [0.01512673, -0.00711861, 0.0, -0.00416198],
        ]
    )


@pytest.fixture(scope="module")
def factor_risk_model_np(factor_exposures, factor_returns, residual_returns):
    return FactorRiskModel(
        factor_exposures=factor_exposures,
        factor_returns=factor_returns,
        residual_returns=residual_returns,
    )


@pytest.fixture(scope="module")
def factor_risk_model(
    factor_exposures,
    factor_returns,
    residual_returns,
    dates,
    instruments,
    factors,
):
    return FactorRiskModel(
        factor_exposures=DataFrame(
            factor_exposures, index=factors, columns=instruments
        ),
        factor_returns=DataFrame(factor_returns, index=dates, columns=factors),
        residual_returns=DataFrame(residual_returns, index=dates, columns=instruments),
    )


def test_cov_estimator_no_adj(factor_risk_model):
    cov_estimator = CovarianceEstimator(factor_risk_model)
    cov = cov_estimator.cov()
    target_cov = DataFrame(
        [
            [0.00038113, 0.00039798, 0.0002858],
            [0.00039798, 0.00068043, 0.00032631],
            [0.0002858, 0.00032631, 0.00048932],
        ],
        index=cov.index,
        columns=cov.columns,
    )
    assert_frame_equal(
        cov,
        target_cov,
    )


def test_cov_estimator_vol_adj(factor_risk_model, instruments):
    cov_estimator = CovarianceEstimator(factor_risk_model)
    vol = Series(0.2, index=instruments)
    cov = cov_estimator.cov(volatility=vol, strict=False)
    target_cov = DataFrame(
        [
            [0.04, 0.03126062, 0.02647162],
            [0.03126062, 0.04, 0.02262061],
            [0.02647162, 0.02262061, 0.04],
        ],
        index=cov.index,
        columns=cov.columns,
    )
    assert_frame_equal(
        cov,
        target_cov,
    )
