import json
from multiprocessing import cpu_count
from multiprocessing.pool import ThreadPool
from os import makedirs
from os.path import join
from typing import Optional

from pandas import DataFrame, Timestamp

from .factor_risk_model import FactorRiskModel
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

            # Skip if the number of sample size is zero
            if y_input.shape[1] == 0:
                continue

            risk_model = self.get(index)
            values[index] = risk_model.transform(
                y=y_input.fillna(0.0),
                regressor=regressor,
            )

        self._values = values
        return self

    def write_directory(
        self, path: str, format: str = "parquet", workers: int = cpu_count(), **kwargs
    ):
        """
        Write to a directory.

        Parameters
        ----------
        path: str
            Directory path to write to.

        format: str
            Supported formats. Default is "parquet". Options
            are "csv", "parquet" and "hdf".

        workers: int
            Number of workers to use for parallel write operations.
            Default is the number of CPUs provided.

        **kwargs: dict
            Optional keyword arguments for the write operation.
        """

        def _frm_write_directory(item):
            key, frm = item
            if isinstance(key, Timestamp):
                key = key.isoformat()
            key_path = join(path, key)
            makedirs(key_path, exist_ok=True)
            frm.write_directory(key_path, format=format, **kwargs)
            return key

        with ThreadPool(processes=workers) as pool:
            keys = pool.map(_frm_write_directory, self._values.items())

        with open(join(path, "metadata.json"), mode="w+") as fp:
            json.dump({"directories": keys, "parameters": self.asdict()}, fp)

    @classmethod
    def read_directory(
        cls, path: str, format: str = "parquet", workers: int = cpu_count(), **kwargs
    ):
        """
        Read a model from directory.

        Parameters
        ----------
        path: str
            Directory path to read from.

        format: str
            Supported formats. Default is "parquet". Options
            are "csv", "parquet" and "hdf".

        workers: int
            Number of workers to use for parallel read operations.
            Default is the number of CPUs provided.

        **kwargs: dict
            Optional keyword arguments for the read operation.
        """

        def _frm_read_directory(key):
            key_path = join(path, key)
            value = FactorRiskModel.read_directory(key_path, format=format, **kwargs)
            return Timestamp(key), value

        with open(join(path, "metadata.json")) as fp:
            metadata = json.load(fp)
            directories = metadata["directories"]
            metadata = metadata["parameters"]

        with ThreadPool(processes=workers) as pool:
            values = pool.map(_frm_read_directory, directories)

        return cls(values=dict(values), **metadata)
