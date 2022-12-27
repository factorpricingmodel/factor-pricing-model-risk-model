from datetime import datetime
from os import makedirs
from os.path import basename, dirname
from os.path import join as fsjoin
from typing import Any, Dict, Optional

import pandas as pd

from ..factor_risk_model import FactorRiskModel
from ..factor_risk_model_transformer import FactorRiskModelTransformer
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


def transform_factor_risk_model(
    risk_model: FactorRiskModel, data: pd.DataFrame, **kwargs
) -> FactorRiskModel:
    transformer = FactorRiskModelTransformer(**kwargs)
    return transformer.transform(risk_model=risk_model, y=data)


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
):
    for key, model in rolling_risk_model.items():
        if not isinstance(key, (pd.Timestamp, datetime)):
            raise TypeError(
                f"Key {key} type must be either datetime / Timestamp, "
                f"rather than {key.__class__.__name__}"
            )
        dump_factor_risk_model(
            risk_model=model,
            success_file=fsjoin(
                dirname(success_file),
                key.isoformat(),
                basename(success_file),
            ),
            format=format,
            parameters=parameters,
        )

    with open(success_file, mode="w") as f:
        f.write("")


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
