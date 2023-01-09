import json
from datetime import datetime
from os import makedirs
from os.path import basename, dirname
from os.path import join as fsjoin
from typing import Any, Dict, Optional

import pandas as pd

from ..factor_risk_model import FactorRiskModel
from ..rolling_factor_risk_model import RollingFactorRiskModel


def generate_factor_risk_model(
    model: str, data: pd.DataFrame, **kwargs
) -> FactorRiskModel:
    """
    Generate factor risk model
    """
    model = model.lower().replace("-", "_")
    if model == "pca":
        from ..statistical.pca import PCA

        model = PCA(**kwargs)
    else:
        raise ValueError(f"Model name {model} is not supported")

    return model.fit(X=data)


def generate_rolling_factor_risk_model(
    model: str, data: pd.DataFrame, model_parameters: Dict[str, Any], **kwargs
) -> RollingFactorRiskModel:
    model = model.lower().replace("-", "_")
    if model == "pca":
        from ..statistical.pca import PCA

        model = PCA(**model_parameters)
    else:
        raise ValueError(f"Model name {model} is not supported")
    rolling_model = RollingFactorRiskModel(model=model, **kwargs)
    return rolling_model.fit(X=data)


def dump_factor_risk_model(
    risk_model: FactorRiskModel,
    success_file: str,
    format: str,
    parameters: Optional[Dict] = None,
):
    """
    Dump factor risk model.
    """
    parameters = parameters or {}
    dumper = f"to_{format}"
    output_directory = dirname(success_file)

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
        name="factor-covariances",
        data=risk_model.factor_covariances,
        output_directory=output_directory,
    )

    _dump(
        name="residual-returns",
        data=risk_model.residual_returns,
        output_directory=output_directory,
    )

    with open(success_file, mode="w") as f:
        f.write("")


def dump_rolling_factor_risk_model(
    rolling_risk_model: RollingFactorRiskModel,
    success_file: str,
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
            success_file=fsjoin(
                dirname(success_file),
                key_name,
                basename(success_file),
            ),
            format=format,
            parameters=parameters,
        )

    with open(success_file, mode="w") as f:
        json.dump({"directories": keys}, f)


def load_factor_risk_model(
    success_file: str,
    format: str,
    parameters: Optional[Dict] = None,
):
    parameters = parameters or {}
    loader = getattr(pd, f"read_{format}")
    output_directory = dirname(success_file)

    def _load(name):
        output_path = fsjoin(output_directory, f"{name}.{format}")
        return loader(output_path, **parameters)

    factor_exposures = _load("factor-exposures")
    factor_returns = _load("factor-returns")
    factor_covariances = _load("factor-covariances")
    residual_returns = _load("residual-returns")
    return FactorRiskModel(
        factor_exposures=factor_exposures,
        factor_returns=factor_returns,
        factor_covariances=factor_covariances,
        residual_returns=residual_returns,
    )


def load_rolling_factor_risk_model(
    success_file: str,
    format: str,
    parameters: Optional[Dict] = None,
    show_progress: Optional[bool] = True,
):
    with open(success_file) as f:
        metadata = json.load(f)
    try:
        directories = metadata["directories"]
    except KeyError:
        raise RuntimeError(
            "Failed to retrieve the list of directories from "
            f"the success file {success_file}. Please ensure "
            "the directory was exported as rolling factor risk "
            "model format"
        )

    if show_progress:
        from tqdm import tqdm

        directories = tqdm(directories, leave=False)

    output_directory = dirname(success_file)
    success_file_name = basename(success_file)
    values = {}
    for directory in directories:
        risk_model_success_file = fsjoin(
            output_directory,
            directory,
            success_file_name,
        )
        risk_model = load_factor_risk_model(
            success_file=risk_model_success_file, format=format, parameters=parameters
        )
        values[pd.Timestamp(directory)] = risk_model

    return RollingFactorRiskModel(
        values=values,
    )


def where_validity(
    validity: pd.DataFrame, data: pd.DataFrame, fillna: Any = None
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

    Returns
    -------
    pd.DataFrame
      Dataframe containing the data for the given universe.
    """
    data = data.reindex_like(validity).where(validity)
    if fillna is not None:
        data = data.fillna(fillna)
    return data
