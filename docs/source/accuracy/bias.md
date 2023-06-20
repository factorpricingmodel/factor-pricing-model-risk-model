# Bias

The objective of a bias test is to examine the forecasting accuracy
of volatility on a time series of observed portfolio returns.

First, standardardized returns $b_t$ are defined as the observed
returns $r_t$ normalised by the volatility forecast $\sigma_t$ deduced
from the risk model

$$
b_t = \frac{r_t}{\sigma_t}
$$

where $r_t$ is the return obtained from the price change in the interval
$[t, t+1]$ and $\sigma_t$ is the volatility forecast of which its risk
model is trained / fitted until time $t$.

Given a rolling window $T$, the bias statistic is defined as

$$
B_T(t) = \sqrt{\frac{1}{T-1} \sum_{\tau=t-T+1}^{t} (b_{\tau} - \bar{b})^2}
$$

The expected bias statistic would be equal to one, while the risk model

- overestimates the risk if it is below one, and
- underestimates the risk otherwise

Assuming the portfolio returns are normally distributed, the 95% confidence
bounds for $B_T(t)$ are between $1-\sqrt{\frac{2}{T}}$ and
$1+\sqrt{\frac{2}{T}}$.

## Usage

Assume that you have market returns `returns` in a `DataFrame`, portfolio
weights `weights` in a `DataFrame` (it could be equally weighted, or market
cap weighted), and rolling risk models, which can be a `RollingRiskModel`
object, or just a dictionary of covariances. To compute the bias statistic
in a 21-day rolling window, you can use the following code snippet.

```
from fpm_risk_model.accuracy.bias import compute_bias_statistics

compute_bias_statistics(
  X=returns,
  weights=weights,
  rolling_risk_model=rolling_risk_model,
  window=30,
)
```

In the meantime, if you have the forecast portfolio volatility, named
`forecast_vols`, from vendor, you can directly pass it into the function
as well.

```
compute_bias_statistics(
  X=returns,
  weights=weights,
  forecast_vols=forecast_vols,
  window=30,
)
```

Please refer to the below section for more information.

## Reference

Alexander, Carol (2009). Market risk analysis, value at risk models. John Wiley & Sons.

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.accuracy.bias
  :members:
```
