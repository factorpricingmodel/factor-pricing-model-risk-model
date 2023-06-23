from datetime import datetime
from typing import Dict, Iterable, Optional, Tuple

from pandas import DataFrame, Timestamp

from .config import Config
from .risk_model import RiskModel


class RollingRiskModelConfig(Config):
    """
    Rollingrisk model configuration.

    Parameters
    ----------
    window: Optional[int]
        Number of rolling windows to use from the returns.
        Must be provided in fitting the model.

    show_progress: Optional[bool]
        Indicate to show progress bar in running.
    """

    window: int
    show_progress: bool = False


class RollingRiskModel:
    """
    Rolling risk model.

    A rolling risk model simply holds a dictionary of risk models
    in a given date / time range. Each key and value pair is a
    date / time and risk model object respectively.
    """

    ConfigClass = RollingRiskModelConfig

    def __init__(
        self,
        model: Optional[RiskModel] = None,
        window: Optional[int] = None,
        show_progress: Optional[bool] = False,
        values: Optional[Dict[datetime, RiskModel]] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        model: Optional[RiskModel]
            Risk model object to fit in rolling basis.

        window: Optional[int]
            Number of rolling windows to use from the returns.
            Must be provided in fitting the model.

        show_progress: Optional[bool]
            Indicate to show progress bar in running.

        values: Optional[Dict[datetime, RiskModel]]
            Rolling risk models values.
        """
        self._model = model
        self._values = values
        self._config = self.ConfigClass(window=window, show_progress=show_progress)

    @property
    def config(self):
        """
        Return the configuration object.
        """
        return self._config

    def get(self, name, **kwargs) -> RiskModel:
        """
        Return a factor risk model from the given name / key.
        """
        try:
            name = Timestamp(name)
        finally:
            return self._values.get(name, **kwargs)

    def keys(self) -> Iterable[datetime]:
        """
        Return a list of keys.
        """
        return self._values.keys()

    def values(self) -> Iterable[RiskModel]:
        """
        Return a list of values.
        """
        return self._values.values()

    def items(self) -> Iterable[Tuple[datetime, RiskModel]]:
        """
        Return a list of tuples with keys and values.
        """
        return self._values.items()

    def fit(
        self,
        X: DataFrame,
        validity: Optional[DataFrame] = None,
        weights: Optional[DataFrame] = None,
    ) -> object:
        """
        Fit the model.

        Parameters
        ----------
        X: DataFrame
            The instrument returns of which its index and columns
            are the date / time and return values.

        validity: DataFrame
            The instrument validity on the date.

        weights: DataFrame
            The weights of the instruments, same dimension as the
            instrument returns.

        Returns
        -------
        object
            The object itself.
        """
        values = {}

        T = X.shape[0]
        iterator = range(0, T)
        if self._config.show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, leave=False)

        try:
            for index in iterator:
                start_index = index
                end_index = index + self._config.window + 1
                if end_index > T:
                    break

                if isinstance(X, DataFrame):
                    X_input = X.iloc[start_index:end_index, :]
                    index_name = X.index[end_index - 1]
                else:
                    raise TypeError(f"Invalid type of X {X.__class__.__name__}")

                if weights is None:
                    weights_input = None
                elif isinstance(weights, DataFrame):
                    weights_input = weights.loc[index_name]
                else:
                    raise TypeError(
                        f"Invalid type of weights {weights.__class__.__name__}"
                    )

                if validity is not None:
                    validity_input = validity.loc[index_name]
                    X_input = X_input.loc[:, validity_input]

                if X_input.shape[1] == 0:
                    continue

                params = {}
                if weights_input is not None:
                    params["weights"] = weights_input

                values[index_name] = self._model.fit(
                    X=X_input.fillna(0.0), **params
                ).copy()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to fit at the index {index} due to error: {exc}"
            )

        self._values = values
        return self

    def asdict(self):
        """
        Returns a dict representation of the object.
        """
        return self._config.dict()
