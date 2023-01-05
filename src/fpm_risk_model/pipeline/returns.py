from typing import Optional, Union

from pandas import DataFrame, Series


def compute_forecast_returns(
    X: Union[DataFrame, Series],
    forecast_timeframe: int,
    min_periods: Optional[int] = None,
) -> Union[DataFrame, Series]:
    """
    Returns the forecast returns for a given forecast timeframe.

    Parameters
    ----------
    X : Union[DataFrame, Series]
        Dataframe of instrument returns.
    forecast_timeframe : int
        The forecast timeframe.
    min_periods : Optional[int]
        Minimum number of observations in the forecast.

    Returns
    ----------
    Union[DataFrame, Series]
        Sum of forecast returns.
    """
    return (
        X.rolling(forecast_timeframe, min_periods=min_periods)
        .mean()
        .shift(-forecast_timeframe)
    )
