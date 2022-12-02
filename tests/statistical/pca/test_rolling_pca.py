import numpy as np
import pandas as pd
import pytest
from numpy import array

from fpm_risk_model.statistical import RollingPCA


@pytest.fixture(scope="module")
def daily_returns_np():
    return array(
        [
            [
                -0.02678756,
                -0.00344077,
                0.00443915,
                -0.04247514,
                -0.01051272,
                -0.01684373,
                0.00658919,
                -0.03482585,
                0.02034743,
                -0.01329412,
            ],
            [
                -0.03400254,
                -0.00953307,
                0.01752232,
                -0.01891826,
                -0.00197782,
                0.01758743,
                0.02239528,
                -0.0452383,
                0.01122229,
                -0.04414332,
            ],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [
                0.000855,
                -0.02505943,
                -0.01956966,
                -0.04220453,
                0.00528776,
                0.01619198,
                0.01451376,
                -0.02571051,
                0.02187115,
                -0.02401548,
            ],
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
        index=instruments,
        columns=dates,
    )


@pytest.fixture(scope="module")
def expected_factor_exposures():
    return {
        5: array(
            [
                [-0.06819719, -0.03899016],
                [-0.09404169, -0.0479342],
                [-0.0, -0.0],
                [-0.08397221, 0.08534767],
            ]
        ),
        6: array(
            [
                [-0.07367893, 0.06325795],
                [-0.08479308, 0.01302083],
                [0.0, -0.0],
                [-0.12127006, -0.04753733],
            ]
        ),
        7: array(
            [
                [-0.09844962, 0.02420275],
                [-0.1369155, 0.03844197],
                [0.0, 0.0],
                [-0.11268921, -0.0678508],
            ]
        ),
        8: array(
            [
                [-0.12167388, 0.04113123],
                [-0.12919982, -0.05521942],
                [-0.0, 0.0],
                [-0.13833261, 0.01539581],
            ]
        ),
        9: array(
            [
                [-0.08217868, -0.06579933],
                [-0.16450881, 0.0291637],
                [-0.0, -0.0],
                [-0.1138375, 0.00535515],
            ]
        ),
    }


@pytest.fixture(scope="module")
def expected_factors():
    return {
        5: array(
            [
                [
                    0.12185252,
                    0.0382722,
                    -0.13415583,
                    0.2808663,
                    -0.09686546,
                    -0.20996973,
                ],
                [
                    0.25302747,
                    -0.13383075,
                    -0.23611538,
                    -0.08802577,
                    0.09167916,
                    0.11326528,
                ],
            ]
        ),
        6: array(
            [
                [
                    0.09850148,
                    -0.03106685,
                    0.30896891,
                    -0.04055805,
                    -0.13259474,
                    -0.20325075,
                ],
                [
                    0.16236479,
                    0.25407535,
                    -0.11386448,
                    -0.11622817,
                    -0.21949329,
                    0.03314581,
                ],
            ]
        ),
        7: array(
            [
                [
                    -0.08097595,
                    0.21458931,
                    -0.04838454,
                    -0.12822228,
                    -0.19570681,
                    0.23870027,
                ],
                [
                    0.29390224,
                    0.14383999,
                    -0.12577298,
                    -0.14691519,
                    -0.01705237,
                    -0.14800168,
                ],
            ]
        ),
        8: array(
            [
                [
                    0.22359551,
                    -0.02619053,
                    -0.09069224,
                    -0.15468593,
                    0.22731175,
                    -0.17933856,
                ],
                [
                    -0.18702111,
                    0.03603959,
                    -0.19957839,
                    -0.06448368,
                    0.2191548,
                    0.19588879,
                ],
            ]
        ),
        9: array(
            [
                [
                    -0.02189233,
                    -0.10611792,
                    -0.16011153,
                    0.24841001,
                    -0.16289345,
                    0.20260521,
                ],
                [
                    0.06104403,
                    0.26177642,
                    -0.00904531,
                    0.09409452,
                    -0.23783891,
                    -0.17003074,
                ],
            ]
        ),
    }


@pytest.fixture(scope="module")
def expected_residual_returns():
    return {
        5: array(
            [
                [
                    7.32481482e-03,
                    9.88799930e-03,
                    2.02071891e-03,
                    -1.08161917e-02,
                    2.39270792e-03,
                    -1.08100492e-02,
                ],
                [
                    -5.52766347e-03,
                    -7.46196782e-03,
                    -1.52493331e-03,
                    8.16242722e-03,
                    -1.80565441e-03,
                    8.15779179e-03,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    2.41735367e-04,
                    3.26326221e-04,
                    6.66882698e-05,
                    -3.56958659e-04,
                    7.89647441e-05,
                    -3.56755943e-04,
                ],
            ]
        ),
        6: array(
            [
                [
                    3.91985187e-03,
                    -3.54810561e-03,
                    -2.13380377e-03,
                    4.22536603e-03,
                    -2.35446801e-03,
                    -1.08840517e-04,
                ],
                [
                    -7.80759631e-03,
                    7.06714876e-03,
                    4.25012959e-03,
                    -8.41612217e-03,
                    4.68965061e-03,
                    2.16789523e-04,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    3.07759134e-03,
                    -2.78572239e-03,
                    -1.67531229e-03,
                    3.31745952e-03,
                    -1.84856229e-03,
                    -8.54539007e-05,
                ],
            ]
        ),
        7: array(
            [
                [
                    0.00495871,
                    -0.00922538,
                    0.00337274,
                    -0.01030656,
                    0.00333949,
                    0.007861,
                ],
                [
                    -0.0034245,
                    0.00637108,
                    -0.00232923,
                    0.00711775,
                    -0.00230626,
                    -0.00542883,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    -0.00017141,
                    0.00031889,
                    -0.00011659,
                    0.00035627,
                    -0.00011544,
                    -0.00027173,
                ],
            ]
        ),
        8: array(
            [
                [
                    0.00537647,
                    -0.00222831,
                    -0.00671623,
                    0.00337372,
                    -0.00322858,
                    0.00342294,
                ],
                [
                    0.00213127,
                    -0.00088332,
                    -0.00266236,
                    0.00133736,
                    -0.00127983,
                    0.00135688,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    -0.00671957,
                    0.00278496,
                    0.00839402,
                    -0.00421651,
                    0.00403512,
                    -0.00427802,
                ],
            ]
        ),
        9: array(
            [
                [
                    -0.00020518,
                    -0.00024968,
                    0.00092623,
                    -0.00013052,
                    -0.00059861,
                    0.00025777,
                ],
                [
                    -0.00066716,
                    -0.00081187,
                    0.00301172,
                    -0.0004244,
                    -0.00194645,
                    0.00083815,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [
                    0.00111225,
                    0.00135349,
                    -0.00502094,
                    0.00070753,
                    0.00324499,
                    -0.00139731,
                ],
            ]
        ),
    }


@pytest.mark.parametrize("speedup", [False, True])
def test_rolling_pca_np(
    daily_returns_np,
    speedup,
    expected_factor_exposures,
    expected_factors,
    expected_residual_returns,
):
    rolling_pca = RollingPCA(
        n_components=2,
        rolling_timeframe=5,
        demean=True,
        speedup=speedup,
    )
    rolling_pca.fit(daily_returns_np)
    for key, factor_exposures in rolling_pca.factor_exposures.items():
        np.testing.assert_almost_equal(
            expected_factor_exposures[key],
            factor_exposures,
        )
    for key, factors in rolling_pca.factors.items():
        np.testing.assert_almost_equal(
            expected_factors[key],
            factors,
        )
    for key, residual_returns in rolling_pca.residual_returns.items():
        np.testing.assert_almost_equal(
            expected_residual_returns[key],
            residual_returns,
        )


@pytest.mark.parametrize("speedup", [False, True])
def test_rolling_pca_pd(
    daily_returns_pd,
    speedup,
    expected_factor_exposures,
    expected_factors,
    expected_residual_returns,
    dates,
    instruments,
):
    rolling_pca = RollingPCA(
        n_components=2,
        rolling_timeframe=5,
        demean=True,
        speedup=speedup,
    )
    rolling_pca.fit(daily_returns_pd)
    for key, factor_exposures in rolling_pca.factor_exposures.items():
        np_key = dates.get_loc(key)
        pd.testing.assert_frame_equal(
            factor_exposures,
            pd.DataFrame(expected_factor_exposures[np_key], index=instruments),
        )
    for key, factors in rolling_pca.factors.items():
        np_key = dates.get_loc(key)
        pd.testing.assert_frame_equal(
            factors,
            pd.DataFrame(
                expected_factors[np_key],
                columns=factors.columns,
            ),
        )
    for key, residual_returns in rolling_pca.residual_returns.items():
        np_key = dates.get_loc(key)
        pd.testing.assert_frame_equal(
            residual_returns,
            pd.DataFrame(
                expected_residual_returns[np_key],
                index=instruments,
                columns=residual_returns.columns,
            ),
        )
