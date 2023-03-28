import pandas as pd
import pytest
from numpy import array


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
