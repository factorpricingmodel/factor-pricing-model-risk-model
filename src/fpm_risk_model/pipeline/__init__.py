import json
from datetime import datetime
from os import makedirs
from os.path import basename, dirname
from os.path import join as fsjoin
from typing import Any, Dict, Optional

import pandas as pd
from pandas import DataFrame

from ..factor_risk_model import FactorRiskModel
from ..rolling_factor_risk_model import RollingFactorRiskModel


def generate_factor_risk_model(
    model: str, data: DataFrame, **kwargs
) -> FactorRiskModel:
    """
    Generate factor risk model

    Parameters
    ----------
    model : str
      Model name supported in statistics module. Supported
      value is `pca`.

    data: DataFrame
      Dataframe of returns of valid instruments, in a dimension
      of (T, N) where N is the number of instruments and T is the
      of timeframes.

    Returns
    -------
    FactorRiskModel
      A fitted factor risk model.
    """
    model = model.lower().replace("-", "_")
    if model == "pca":
        from ..statistical.pca import PCA

        model = PCA(**kwargs)
    elif model == "apca":
        from ..statistical.apca import APCA

        model = APCA(**kwargs)
    else:
        raise ValueError(f"Model name {model} is not supported")

    return model.fit(X=data)


def generate_rolling_factor_risk_model(
    model: str,
    data: DataFrame,
    model_parameters: Dict[str, Any],
    weights: Optional[DataFrame] = None,
    **kwargs,
) -> RollingFactorRiskModel:
    model = model.lower().replace("-", "_")
    if model == "pca":
        from ..statistical.pca import PCA

        model = PCA(**model_parameters)
    elif model == "apca":
        from ..statistical.apca import APCA

        model = APCA(**model_parameters)
    else:
        raise ValueError(f"Model name {model} is not supported")
    rolling_model = RollingFactorRiskModel(model=model, **kwargs)

    params = {}
    if weights is not None:
        params["weights"] = weights

    return rolling_model.fit(X=data, **params)


def dump_factor_risk_model(
    risk_model: FactorRiskModel,
    metadata_file: str,
    format: str,
    parameters: Optional[Dict] = None,
):
    """
    Dump factor risk model.
    """
    parameters = parameters or {}
    dumper = f"to_{format}"
    output_directory = dirname(metadata_file)

    def _dump(name, data, output_directory):
        if isinstance(data, pd.DataFrame):
            makedirs(output_directory, exist_ok=True)
            getattr(data, dumper)(
                fsjoin(output_directory, f"{name}.{format}"), **parameters
            )
        elif isinstance(data, dict):
            for key, value in data.items():
                _dump(
                    name=key,
                    data=value,
                    output_directory=fsjoin(output_directory, name),
                )
        else:
            raise TypeError(f"Unrecognised type {data.__class__.__class__} to export")

    _dump(
        name="factor-exposures",
        data=risk_model.factor_exposures,
        output_directory=output_directory,
    )

    _dump(
        name="factor-returns",
        data=risk_model.factor_returns,
        output_directory=output_directory,
    )

    _dump(
        name="residual-returns",
        data=risk_model.residual_returns,
        output_directory=output_directory,
    )

    with open(metadata_file, mode="w") as f:
        f.write(json.dumps({"parameters": risk_model.asdict()}))


def dump_rolling_factor_risk_model(
    rolling_risk_model: RollingFactorRiskModel,
    metadata_file: str,
    format: str,
    parameters: Optional[Dict] = None,
    show_progress: Optional[bool] = True,
):
    keys = []
    iterator = rolling_risk_model.items()
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(iterator, leave=False)
    for key, model in iterator:
        if not isinstance(key, (pd.Timestamp, datetime)):
            raise TypeError(
                f"Key {key} type must be either datetime / Timestamp, "
                f"rather than {key.__class__.__name__}"
            )
        key_name = key.isoformat()
        keys.append(key_name)
        dump_factor_risk_model(
            risk_model=model,
            metadata_file=fsjoin(
                dirname(metadata_file),
                key_name,
                basename(metadata_file),
            ),
            format=format,
            parameters=parameters,
        )

    with open(metadata_file, mode="w") as f:
        f.write(
            json.dumps({"directories": keys, "parameters": rolling_risk_model.asdict()})
        )


def load_factor_risk_model(
    metadata_file: str,
    format: str,
    parameters: Optional[Dict] = None,
):
    parameters = parameters or {}
    loader = getattr(pd, f"read_{format}")
    output_directory = dirname(metadata_file)

    def _load(name):
        output_path = fsjoin(output_directory, f"{name}.{format}")
        return loader(output_path, **parameters)

    factor_exposures = _load("factor-exposures")
    factor_returns = _load("factor-returns")
    residual_returns = _load("residual-returns")
    return FactorRiskModel(
        factor_exposures=factor_exposures,
        factor_returns=factor_returns,
        residual_returns=residual_returns,
    )


def load_rolling_factor_risk_model(
    metadata_file: str,
    format: str,
    parameters: Optional[Dict] = None,
    show_progress: Optional[bool] = True,
):
    with open(metadata_file) as f:
        metadata = json.load(f)
    try:
        directories = metadata["directories"]
    except KeyError:
        raise RuntimeError(
            "Failed to retrieve the list of directories from "
            f"the metadata file {metadata_file}. Please ensure "
            "the directory was exported as rolling factor risk "
            "model format"
        )

    if show_progress:
        from tqdm import tqdm

        directories = tqdm(directories, leave=False)

    output_directory = dirname(metadata_file)
    metadata_file_name = basename(metadata_file)
    values = {}
    for directory in directories:
        risk_model_metadata_file = fsjoin(
            output_directory,
            directory,
            metadata_file_name,
        )
        risk_model = load_factor_risk_model(
            metadata_file=risk_model_metadata_file, format=format, parameters=parameters
        )
        values[pd.Timestamp(directory)] = risk_model

    model_parameters = metadata.get("parameters", {})
    return RollingFactorRiskModel(values=values, **model_parameters)


def where_validity(
    validity: pd.DataFrame,
    data: pd.DataFrame,
    fillna: Any = None,
    ffill: Optional[bool] = False,
) -> pd.DataFrame:
    """
    Return the data for the given universe.

    Parameters
    ----------
    validity : pd.DataFrame
      Validity of the universe of which the index and columns are date / time
      and instrument names respectively.
    data: pd.DataFrame
      Data of which the index and columns are date / time and instrument names
      respectively.
    fillna: Any
      Handle nan values which includes data outside of the universe.
    ffill: Optional[bool]
      Indicates to forward fill the data. Default is `False`.

    Returns
    -------
    pd.DataFrame
      Dataframe containing the data for the given universe.
    """
    data = data.reindex_like(validity).where(validity)
    if ffill:
        data = data.ffill()
    if fillna is not None:
        data = data.fillna(fillna)
    return data
