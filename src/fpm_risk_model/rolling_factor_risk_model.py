from typing import Optional

from pandas import DataFrame

from .rolling_risk_model import RollingRiskModel


class RollingFactorRiskModel(RollingRiskModel):
    """
    Rolling factor risk model.

    Rolling factor risk model is a subclass of rolling risk model,
    while it supports transforming each risk model given the instrument
    returns.
    """

    def __init__(self, **kwargs):
        """
        Constructor.
        """
        super().__init__(**kwargs)

    def transform(
        self,
        y: DataFrame,
        regressor: Optional[object] = None,
    ) -> object:
        """
        Transform the rolling factor risk model.

        The method is used to transform the rolling factor risk model by
        passing another set of returns. Most of the time, the
        factor risk model is fitted by the estimation universe,
        and then transformed by the model universe.

        Parameters
        ----------
        y : DataFrame
            The instrument returns of which its index and columns
            are the date / time and return values.

        regressor : object, default=None
            Regressor to transform the input y into factor exposures.
            If None, the regressor is set to the default WLS.

        Returns
        -------
        object
            The transformed rolling factor risk model.
        """
        if not isinstance(y, DataFrame):
            raise TypeError(
                "Only DataFrame type is supported, but not " f"{y.__class__.__name__}"
            )

        if not self._window:
            raise ValueError(
                f"Rolling timeframe must be specified, but not {self._window}"
            )

        T = y.shape[0]
        values = {}
        iterator = range(T)
        if self._show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, leave=False)

        for index in iterator:
            start_index = index
            end_index = index + self._window + 1
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

        self._values = values
        return self
