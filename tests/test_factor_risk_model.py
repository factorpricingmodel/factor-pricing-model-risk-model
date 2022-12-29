import pytest

from numpy import array, nan
import numpy as np
from pandas import DataFrame
import pandas as pd

from fpm_risk_model.factor_risk_model import FactorRiskModel


@pytest.fixture(scope="module")
def instruments():
    return ["A", "AAL", "AAP", "AAPL"]


@pytest.fixture(scope="module")
def dates():
    return pd.bdate_range("2016-01-04", "2016-01-15")


@pytest.fixture(scope="module")
def factors():
    return ["factor_1", "factor_2"]


@pytest.fixture(scope="module")
def daily_returns_np():
    return array(
        [
            [-0.02678756, -0.03400254, 0.0, 0.000855],
            [-0.00344077, -0.00953307, 0.0, -0.02505943],
            [0.00443915, 0.01752232, 0.0, -0.01956966],
            [-0.04247514, -0.01891826, 0.0, -0.04220453],
            [-0.01051272, -0.00197782, 0.0, 0.00528776],
            [-0.01684373, 0.01758743, 0.0, 0.01619198],
            [0.00658919, 0.02239528, 0.0, 0.01451376],
            [-0.03482585, -0.0452383, 0.0, -0.02571051],
            [0.02034743, 0.01122229, 0.0, 0.02187115],
            [-0.01329412, -0.04414332, 0.0, -0.02401548],
        ]
    )


@pytest.fixture(scope="module")
def daily_returns_pd(daily_returns_np, instruments, dates):
    return DataFrame(
        daily_returns_np,
        index=dates,
        columns=instruments,
    )


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
def factor_covariances():
    return array([[1.11111111e-02, -1.13074741e-18], [-1.13074741e-18, 1.11111111e-02]])


@pytest.fixture(scope="module")
def factor_risk_model_np(
    factor_exposures, factor_returns, factor_covariances, residual_returns
):
    return FactorRiskModel(
        factor_exposures=factor_exposures,
        factor_returns=factor_returns,
        factor_covariances=factor_covariances,
        residual_returns=residual_returns,
    )


@pytest.fixture(scope="module")
def factor_risk_model_pd(
    factor_exposures,
    factor_returns,
    factor_covariances,
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
        factor_covariances=DataFrame(
            factor_covariances, index=factors, columns=factors
        ),
        residual_returns=DataFrame(residual_returns, index=dates, columns=instruments),
    )


@pytest.fixture(scope="module")
def expected_covariances():
    return array(
        [
            [0.00038113, 0.00039798, nan, 0.0002858],
            [0.00039798, 0.00068043, nan, 0.00032631],
            [nan, nan, nan, nan],
            [0.0002858, 0.00032631, nan, 0.00048932],
        ]
    )


@pytest.fixture(scope="module")
def expected_correlations():
    return array(
        [
            [1.0, 0.78151541, nan, 0.66179042],
            [0.78151541, 1.0, nan, 0.56551526],
            [nan, nan, nan, nan],
            [0.66179042, 0.56551526, nan, 1.0],
        ]
    )


def test_factor_risk_model_np_covariances(factor_risk_model_np, expected_covariances):
    cov = factor_risk_model_np.cov()
    np.testing.assert_allclose(cov, expected_covariances, atol=1e-7)


def test_factor_risk_model_np_correlations(factor_risk_model_np, expected_correlations):
    corr = factor_risk_model_np.corr()
    np.testing.assert_allclose(corr, expected_correlations)


def test_factor_risk_model_pd_covariances(
    factor_risk_model_pd, expected_covariances, instruments
):
    cov = factor_risk_model_pd.cov()
    expected_covariances = pd.DataFrame(
        expected_covariances, index=instruments, columns=instruments
    )
    pd.testing.assert_frame_equal(
        cov,
        expected_covariances,
    )


def test_factor_risk_model_pd_correlations(
    factor_risk_model_pd, expected_correlations, instruments
):
    corr = factor_risk_model_pd.corr()
    expected_correlations = pd.DataFrame(
        expected_correlations, index=instruments, columns=instruments
    )
    pd.testing.assert_frame_equal(corr, expected_correlations)
