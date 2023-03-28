import numpy as np
import pandas as pd
import pytest
from numpy import array

from fpm_risk_model.statistical import APCA


@pytest.fixture(scope="module")
def expected_factor_exposures():
    return array([[0.0182826, 0.0667684, 0.0, 0.0081343]])


@pytest.fixture(scope="module")
def expected_factor_returns():
    return array(
        [
            [-0.42642457],
            [0.06761142],
            [0.45389456],
            [0.24108103],
            [0.00924802],
            [0.30078892],
            [0.27190434],
            [-0.37970421],
            [-0.05028576],
            [-0.48811374],
        ]
    )


@pytest.fixture(scope="module")
def expected_residual_returns():
    return array(
        [
            [-0.00731101, 0.00297776, 0.0, 0.01210767],
            [0.00700353, -0.00553878, 0.0, -0.01782541],
            [0.0078212, -0.00427491, 0.0, -0.01547779],
            [-0.03520231, -0.02650626, 0.0, -0.03638156],
            [0.00099861, 0.0059133, 0.0, 0.01299653],
            [-0.01066251, 0.00601283, 0.0, 0.02152926],
            [0.01329849, 0.01274925, 0.0, 0.020086],
            [-0.01620347, -0.01137745, 0.0, -0.01483788],
            [0.0329472, 0.02308839, 0.0, 0.03006419],
            [0.00731027, -0.00304413, 0.0, -0.01226101],
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
