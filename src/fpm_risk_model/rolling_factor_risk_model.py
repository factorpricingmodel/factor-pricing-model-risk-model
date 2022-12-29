from typing import Iterable, Optional, Tuple

import pandas as pd
from pandas import Timestamp

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

    def get(self, name) -> FactorRiskModel:
        """
        Return a factor risk model from the given name / key.
        """
        return self._values.get(name)

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

    def fit(self, X) -> object:
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
