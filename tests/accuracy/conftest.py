import pytest
from pandas import DataFrame, bdate_range

from fpm_risk_model.rolling_factor_risk_model import RollingFactorRiskModel
from fpm_risk_model.statistical import PCA


@pytest.fixture(scope="module")
def instruments():
    return ["A", "AAL", "AAP", "AAPL"]


@pytest.fixture(scope="module")
def dates():
    return bdate_range("2016-01-04", "2016-01-15")


@pytest.fixture(scope="module")
def factors():
    return [
        "factor_1",
        "factor_2",
    ]


@pytest.fixture(scope="module")
def daily_returns(instruments, dates):
    return DataFrame(
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
        ],
        columns=instruments,
        index=dates,
    )


@pytest.fixture(scope="module")
def weights(instruments, dates):
    return DataFrame(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
            [0.6, 0.2, 0.0, 0.2],
            [0.6, 0.2, 0.0, 0.2],
            [0.6, 0.2, 0.0, 0.2],
        ],
        columns=instruments,
        index=dates,
    )


@pytest.fixture(scope="module")
def rolling_factor_risk_model(daily_returns):
    model = PCA(
        n_components=2,
        demean=True,
        speedup=True,
    )
    rolling_model = RollingFactorRiskModel(
        model=model,
        rolling_timeframe=5,
        show_progress=False,
    )
    return rolling_model.fit(X=daily_returns)
