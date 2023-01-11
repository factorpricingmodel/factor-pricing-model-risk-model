# Risk Model

Risk model is an abstract class to produce risk metrics. Pairwise
covariance and other related metrics, like correlation, are
supported.

## Fit the model

In general, all the risk models are fitted by passing the instrument
returns, It means after initialising the risk model object, call
the method `fit(X=...)`.

The method returns the object itself while the object is already
stored with fitted attributes.

To keep the risk model fitted attributes, you should simply call
the method `copy()` to ensure a deep clone of the fitted result.

## Rolling risk model

Rolling risk model fits risk models at each time t given all the
information provided at or before time t. Primarily rolling risk
models are fitted in a rolling window of information, for example,
instrument returns.

In technical point of view, rolling risk model object holds a
dictionary of risk models and its keys are date / time. A
`pandas.DataFrame` of instrument returns in ascending time series
order should be passed into the method `fit` to ensure only
retrospective information is used, rather than future one.

For example, to construct a rolling risk model fitted by PCA
model in a rolling window of 63 days (3 business months equivalent)

```
from fpm_risk_model import RollingRiskModel
from fpm_risk_model.statistics.pca import PCA

rolling_risk_model = RollingRiskModel(
  model=PCA(n_components=5),
  rolling_timeframes=63,
)

# Instrument returns must be a DataFrame
rolling_risk_model.fit(instrument_returns)
```

To retrieve a risk model, call method `get` with a specified
date

```
from pandas import Timestamp
rolling_risk_model.get(Timestamp("2000-01-01"))
```

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.risk_model
  :members:
```

```{eval-rst}
.. automodule:: fpm_risk_model.rolling_risk_model
  :members:
```
