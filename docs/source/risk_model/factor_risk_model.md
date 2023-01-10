# Factor Risk Model

A factor risk model is a statistical model that is used to quantify the risk of an investment or portfolio. It takes into account one or more risk factors, or variables that are known to affect the risk of the investment. The model aims to identify the factors that contribute to risk and to measure their impact on the investment's return.

## Description

A factor risk model describes the instrument / portfolio returns $R$ by factor
exposures $B$, factor returns $F$ and residual returns (sometimes named as
idiosyncratic returns) $U$.

$$
R = BF + U
$$

Therefore, a factor risk model object contains the following attributes

- Factor returns: The time series of factor returns, in dimension of (T, n)

- Factor covariance: The covariance of the factors derived from the factor
  returns, in dimension of (n, n)

- Factor exposures: The exposure of the instrument / portfolio on each factor,
  in dimension of (n, N)

- Residual returns: The residual returns specific for each instrument /
  portfolio, in dimension of (T, N)

where T, N and n are the number of timeframes, instruments and factors.

## Transform

Transformation allows the risk model to be
[expanded](https://factor-pricing-model-risk-model.readthedocs.io/en/latest/risk_model/universe.html)
from estimation universe to model universe.

Transformation requires passing the instrument returns of the model
universe in

1. Same length and granularity of time series
2. Homogeneity

and the factor returns to examine their exposures and residual returns.

One usage is to transform the trained risk model to a bigger universe

```
risk_model.fit(estimation_returns)

transformed_risk_model = risk_model.transform(model_returns)
```

Another usage is to combine the factor returns from risk models derived
from different methodologies, e.g. statistical and fundamental, and
then transform into a more generic risk model.

For example, `model1` and `model2` contain the statistical and fundamental
factors respectively. To combine them and then transform into a new risk
model

```
import pandas as pd

risk_model = FactorRiskModel(
    factor_returns=pd.concat([model1.factor_returns, model2.factor_returns])
)
risk_model.transform(returns)
```

The transformed risk model is always updated in place. To retain the original
risk model, please always use `copy` as a backup.

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.factor_risk_model
  :members:
```
