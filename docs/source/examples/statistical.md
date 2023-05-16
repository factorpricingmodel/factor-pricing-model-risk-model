# Example: Statistical factor risk model

## Objective

The following example shows how to use the library
`factor-pricing-model-universe` and `factor-pricing-model-risk-model`
to build a statistical factor risk model and benchmark
with other models provided by, for example, vendors.

## Data

First, we need daily data on price, volume and marketcap to construct
the universe. (TBC)

## Building the universe

Given that the price, volume and marketcap data are provided,
we can then build the estimation and model universe.

|        Parameter         |    Estimation    |      Model       |
| :----------------------: | :--------------: | :--------------: |
|    Marketcap ranking     |     Top 90%      |     Top 50%      |
| Rolling 63-day liquidity | 90% availability | 90% availability |

The following graph shows the number of instruments in U.S.
equities in the major indices.

## Build first PCA factor model

We can now start building the first PCA factor model for testing.
First, create a PCA model with 20 components. Second, pass
the daily returns of adjusted close prices of the estimation
universe (selecting only estimation universe returns by `where`).
Fill the missing returns by 0.0, especially when the particular
instruments were halted trading. Finally, fit the PCA model
with the given daily returns.

```
from fpm_risk_model.statistics import PCA

estimation_returns = (
    adjusted_close_returns
    .where(estimation_validity)
    .fillna(0.0)
)
pca = PCA(n_components=20)
pca.fit(estimation_returns)
```

We can then examine the factor returns and exposures on the fitted
model.

```
pca.factor_returns
pca.factor_exposures
```

And also the pairwise covariance of the estimation universe.

```
pca.cov()
```

Once the model is fit, we can transform the model with
a (larger) model universe.

```
transformed_pca = pca.transform(model_returns)
```

And, again, its pairwise covariance can be estimated

```
transformed_pca.cov()
```

The covariance contains most instruments in the whole
universe scope.

## Evaluation on rolling basis

The above example returns a factor model using the full history
of daily returns. However, in practical usage, the factor model
is evaluated in a rolling basis.

```
from fpm_risk_model import RollingFactorRiskModel
from fpm_risk_model.statistics import PCA

pca = PCA(n_components=20)
rolling_pca = RollingFactorRiskModel(model=pca, window=252)
rolling_pca.fit(estimation_returns)
```

Similarly, the rolling factor model can be transformed
with model universe, and the pairwise covariance can be
retrieved given a specified date.

```
rolling_pca = rolling_pca.transform(model_returns)
rolling_pca.get("2019-06-13").cov()
```

## Evaluation

Once the risk model is built, we may want to evaluate its
accuracy, comparing with the existing risk models, or
risk models provided by vendors. Currently, there are
two metrics are provided in the library for accuracy
evaluation.

- Bias
- Value-at-Risk (VaR)
