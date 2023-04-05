import numpy as np
import pandas as pd
import pytest
from numpy import array

from fpm_risk_model.statistical import APCA


@pytest.fixture(scope="module")
def expected_factor_exposures():
    return array([[0.01162204, 0.04715992, 0.0, -0.01902376]])


@pytest.fixture(scope="module")
def expected_factor_returns():
    return array(
        [
            [-0.50982849],
            [0.2321729],
            [0.56366303],
            [0.36835974],
            [-0.09223446],
            [0.05441901],
            [0.13598935],
            [-0.28669822],
            [-0.15512892],
            [-0.31071395],
        ]
    )


@pytest.fixture(scope="module")
def expected_residual_returns():
    return array(
        [
            [-0.0091819, -0.00145047, 0.0, -0.00105986],
            [0.00554132, -0.01197373, 0.0, -0.01285863],
            [0.00956865, -0.00055138, 0.0, -0.00106267],
            [-0.03507582, -0.02778148, 0.0, -0.02741295],
            [0.00223964, 0.01088055, 0.0, 0.01131711],
            [-0.00579578, 0.02352963, 0.0, 0.02501123],
            [0.01668913, 0.02449063, 0.0, 0.02488478],
            [-0.01981342, -0.02320904, 0.0, -0.02338059],
            [0.03383076, 0.02704676, 0.0, 0.02670401],
            [0.00199742, -0.02098148, 0.0, -0.02214243],
        ]
    )


def test_apca_np(
    daily_returns_np,
    expected_factor_exposures,
    expected_factor_returns,
    expected_residual_returns,
):
    apca = APCA(n_components=1, demean=True)
    apca.fit(X=daily_returns_np)
    np.testing.assert_almost_equal(
        apca.factor_exposures,
        expected_factor_exposures,
    )
    np.testing.assert_almost_equal(
        apca.factor_returns,
        expected_factor_returns,
    )
    np.testing.assert_almost_equal(
        apca.residual_returns,
        expected_residual_returns,
    )


def test_apca_pd(
    daily_returns_pd,
    expected_factor_exposures,
    expected_factor_returns,
    expected_residual_returns,
    instruments,
    dates,
):
    apca = APCA(n_components=1, demean=True)
    apca.fit(X=daily_returns_pd)

    pd.testing.assert_frame_equal(
        apca.factor_exposures,
        pd.DataFrame(
            expected_factor_exposures,
            index=apca.factor_exposures.index,
            columns=instruments,
        ),
    )
    pd.testing.assert_frame_equal(
        apca.factor_returns,
        pd.DataFrame(
            expected_factor_returns,
            index=dates,
            columns=apca.factor_returns.columns,
        ),
    )
    pd.testing.assert_frame_equal(
        apca.residual_returns,
        pd.DataFrame(
            expected_residual_returns,
            columns=instruments,
            index=dates,
        ),
    )
