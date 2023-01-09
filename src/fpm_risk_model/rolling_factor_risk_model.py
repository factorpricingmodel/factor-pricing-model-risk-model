from typing import Iterable, Optional, Tuple

import pandas as pd
from pandas import DataFrame, Timestamp

from .factor_risk_model import FactorRiskModel


class RollingFactorRiskModel:
    """
    Rolling factor risk model.
    """

    def __init__(
        self,
        model: Optional[FactorRiskModel] = None,
        rolling_timeframe: Optional[int] = None,
        show_progress: Optional[bool] = False,
        values: Optional[object] = None,
    ):
        self._model = model
        self._rolling_timeframe = rolling_timeframe
        self._show_progress = show_progress
        self._values = values

    def get(self, name, **kwargs) -> FactorRiskModel:
        """
        Return a factor risk model from the given name / key.
        """
        return self._values.get(name, **kwargs)

    def keys(self) -> Iterable[object]:
        """
        Return a list of keys.
        """
        return self._values.keys()

    def values(self) -> Iterable[object]:
        """
        Return a list of values.
        """
        return self._values.values()

    def items(self) -> Iterable[Tuple[Timestamp, FactorRiskModel]]:
        """
        Return a list of tuples with keys and values.
        """
        return self._values.items()

    def fit(self, X: DataFrame) -> object:
        """
        Fit the model.
        """
        values = {}

        T = X.shape[0]
        iterator = range(0, T)
        if self._show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, leave=False)

        try:
            for index in iterator:
                start_index = index
                end_index = index + self._rolling_timeframe + 1
                if end_index > T:
                    break

                if isinstance(X, pd.DataFrame):
                    X_input = X.iloc[start_index:end_index, :]
                    index_name = X.index[end_index - 1]
                else:
                    raise TypeError(f"Invalid type of X {X.__class__.__name__}")
                values[index_name] = self._model.fit(X=X_input).copy()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fit at the index {index} due to error: {exc}"
            )

        self._values = values
        return self

    def transform(
        self,
        y: DataFrame,
        regressor: Optional[object] = None,
        rolling_timeframe: Optional[int] = None,
    ) -> object:
        """
        Transform the rolling factor risk model.

        The method is used to transform the rolling factor risk model by
        passing another set of returns. Most of the time, the
        factor risk model is fitted by the estimation universe,
        and then transformed by the model universe.

        Parameters
        ----------
        y : ndarray
            The instrument returns.

        regressor : object, default=None
            Regressor to transform the input y into factor exposures.
            If None, the regressor is set to the default WLS.

        rolling_timeframe : int, default=None
            The rolling timeframe.

        Returns
        -------
        object
            The transformed rolling factor risk model.
        """
        if not isinstance(y, DataFrame):
            raise TypeError(
                "Only DataFrame type is supported, but not " f"{y.__class__.__name__}"
            )

        rolling_timeframe = rolling_timeframe or self._rolling_timeframe
        if not rolling_timeframe:
            raise ValueError(
                f"Rolling timeframe must be specified, but not {rolling_timeframe}"
            )

        T = y.shape[0]
        values = {}
        iterator = range(T)
        if self._show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, leave=False)

        for index in iterator:
            start_index = index
            end_index = index + rolling_timeframe + 1
            if end_index > T:
                break

            y_input = y.iloc[start_index:end_index, :]
            index_name = y.index[end_index - 1]

            if index_name not in self.keys():
                raise ValueError(
                    f"Index {index_name} cannot be found in the given "
                    "risk model. The risk model cannot be transformed "
                    "by the given returns"
                )

            risk_model = self.get(index_name)
            values[index_name] = risk_model.transform(
                y=y_input,
                regressor=regressor,
            )

        return RollingFactorRiskModel(
            values=values,
            rolling_timeframe=rolling_timeframe,
        )
