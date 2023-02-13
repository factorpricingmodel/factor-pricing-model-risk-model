import numpy as np
import pandas as pd
import pytest
from numpy import array

from fpm_risk_model.statistical import PCA


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
def instruments():
    return ["A", "AAL", "AAP", "AAPL"]


@pytest.fixture(scope="module")
def dates():
    return pd.bdate_range("2016-01-04", "2016-01-15")


@pytest.fixture(scope="module")
def daily_returns_pd(daily_returns_np, instruments, dates):
    return pd.DataFrame(
        daily_returns_np,
        index=dates,
        columns=instruments,
    )


@pytest.fixture(scope="module")
def expected_factor_exposures():
    return array(
        [
            [-0.15454215, -0.22795166, 0.0, -0.17179763],
            [0.00706732, 0.08354979, 0.0, -0.11721647],
        ]
    )


@pytest.fixture(scope="module")
def expected_factor_returns():
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
def expected_residual_returns():
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


@pytest.mark.parametrize("speedup", [True, False])
def test_pca_np(
    daily_returns_np,
    speedup,
    expected_factor_exposures,
    expected_factor_returns,
    expected_residual_returns,
):
    pca = PCA(n_components=2, demean=True, speedup=speedup)
    pca.fit(X=daily_returns_np)
    np.testing.assert_almost_equal(
        pca.factor_exposures,
        expected_factor_exposures,
    )
    np.testing.assert_almost_equal(
        pca.factor_returns,
        expected_factor_returns,
    )
    np.testing.assert_almost_equal(
        pca.residual_returns,
        expected_residual_returns,
    )


@pytest.mark.parametrize("speedup", [False, True])
def test_pca_pd(
    daily_returns_pd,
    speedup,
    expected_factor_exposures,
    expected_factor_returns,
    expected_residual_returns,
    instruments,
    dates,
):
    pca = PCA(n_components=2, demean=True, speedup=speedup)
    pca.fit(X=daily_returns_pd)

    pd.testing.assert_frame_equal(
        pca.factor_exposures,
        pd.DataFrame(
            expected_factor_exposures,
            index=pca.factor_exposures.index,
            columns=instruments,
        ),
    )
    pd.testing.assert_frame_equal(
        pca.factor_returns,
        pd.DataFrame(
            expected_factor_returns,
            index=dates,
            columns=pca.factor_returns.columns,
        ),
    )
    pd.testing.assert_frame_equal(
        pca.residual_returns,
        pd.DataFrame(
            expected_residual_returns,
            columns=instruments,
            index=dates,
        ),
    )


@pytest.mark.parametrize("speedup", [False, True])
def test_pca_same_covariances(
    daily_returns_pd,
    speedup,
):
    """
    Covariances should be the same if the number of components
    is same as the rank of the daily returns.
    """
    factor_risk_model = PCA(n_components=3, speedup=speedup, show_all_instruments=True)
    factor_risk_model.fit(X=daily_returns_pd)
    expected_covariances = daily_returns_pd.cov()
    pd.testing.assert_frame_equal(
        expected_covariances,
        factor_risk_model.cov().fillna(0.0),
    )
