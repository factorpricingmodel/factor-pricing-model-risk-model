from numpy import array
import numpy as np
import pandas as pd

import pytest

from fpm_risk_model.factor_risk_model import FactorRiskModel
from fpm_risk_model.factor_risk_model_transformer import FactorRiskModelTransformer


@pytest.fixture(scope="module")
def daily_returns_np():
    return array(
        [
            [-0.02678756, -0.03400254, 0.01149437, 0.000855, -0.02751516],
            [-0.00344077, -0.00953307, -0.00683127, -0.02505943, -0.00416587],
            [0.00443915, 0.01752232, -0.02645521, -0.01956966, 0.00017417],
            [-0.04247514, -0.01891826, 0.01107375, -0.04220453, -0.00296276],
            [-0.01051272, -0.00197782, -0.02197155, 0.00528776, -0.02726779],
            [-0.01684373, 0.01758743, 0.0102368, 0.01619198, -0.0318059],
            [0.00658919, 0.02239528, 0.00693622, 0.01451376, 0.01781705],
            [-0.03482585, -0.0452383, -0.03991384, -0.02571051, -0.05693729],
            [0.02034743, 0.01122229, -0.00590854, 0.02187115, 0.06604132],
            [-0.01329412, -0.04414332, 0.02101588, -0.02401548, 0.050953],
        ]
    )


@pytest.fixture(scope="module")
def instruments():
    return ["A", "AAL", "AAP", "AAPL", "ABBV"]


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
def factor_covariances():
    return array([[1.11111111e-02, -1.13074741e-18], [-1.13074741e-18, 1.11111111e-02]])


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
def factor_risk_model(
    factor_exposures,
    factor_returns,
    factor_covariances,
    residual_returns,
):
    return FactorRiskModel(
        factor_exposures=factor_exposures,
        factor_returns=factor_returns,
        factor_covariances=factor_covariances,
        residual_returns=residual_returns,
    )


def test_factor_risk_model_transformer(daily_returns_np, factor_risk_model):
    transformer = FactorRiskModelTransformer()
    transformed_model = transformer.transform(
        y=daily_returns_np, risk_model=factor_risk_model
    )

    # No change on factor returns and factor covariances
    np.testing.assert_almost_equal(
        factor_risk_model.factor_returns,
        transformed_model.factor_returns,
    )
    np.testing.assert_almost_equal(
        factor_risk_model.factor_covariances,
        transformed_model.factor_covariances,
    )
    # Factor exposures of the estimation universe should be the same
    np.testing.assert_almost_equal(
        transformed_model.factor_exposures,
        array(
            [
                [-0.15454215, -0.22795167, -0.00057977, -0.17179763, -0.12538096],
                [0.00706732, 0.08354978, -0.03289704, -0.11721647, 0.01247999],
            ]
        ),
    )
    # Residual returns
    np.testing.assert_almost_equal(
        transformed_model.residual_returns,
        array(
            [
                [-0.01591017, -0.00651808, 0.00638443, -0.00662022, -0.01763485],
                [-0.00129242, -0.01339716, -0.00365674, -0.01064216, -0.00307173],
                [-0.00619755, -0.01108882, -0.02067724, -0.00929256, -0.00964756],
                [-0.024343, -0.00254961, 0.01581362, -0.0043, 0.01079066],
                [-0.01592217, -0.00651244, -0.02354262, -0.00661692, -0.03133874],
                [-0.03021377, 0.00021316, 0.00913007, -0.00268471, -0.0424366],
                [-0.01353733, -0.00763474, 0.00701521, -0.00727308, 0.00145663],
                [-0.01239597, -0.00817186, -0.04162256, -0.00758712, -0.03837243],
                [-0.00043806, -0.01379923, -0.00852465, -0.01087723, 0.049698],
                [0.00344632, -0.01562721, 0.0193571, -0.01194598, 0.06488738],
            ]
        ),
    )
