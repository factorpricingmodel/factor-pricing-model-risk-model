from typing import Optional

from pandas import DataFrame, Timestamp

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
        validity: Optional[DataFrame] = None,
        regressor: Optional[object] = None,
        start_date: Optional[Timestamp] = None,
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

        validity: DataFrame
            The instrument validity on the date.

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

        if not self._config.window:
            raise ValueError(
                f"Rolling timeframe must be specified, but not {self._config.window}"
            )

        values = {}
        iterator = self.keys()
        if self._config.show_progress:
            from tqdm import tqdm

            iterator = tqdm(iterator, leave=False)

        for index in iterator:
            if start_date is not None and start_date > index:
                continue
            y_end_index = y.index.get_loc(index)
            y_start_index = y_end_index - self._config.window
            if y_start_index < 0:
                raise ValueError(
                    "Input data does not have sufficient history for " f"index {index}"
                )

            y_input = y.iloc[y_start_index : y_end_index + 1]
            if validity is not None:
                validity_input = validity.loc[index]
                y_input = y_input.loc[:, validity_input]

            risk_model = self.get(index)
            values[index] = risk_model.transform(
                y=y_input.fillna(0.0),
                regressor=regressor,
            )

        self._values = values
        return self
