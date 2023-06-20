# Value at risk (VaR)

Value at risk (VaR) is a risk metric representing the maximum possible
loss for a given holding horizon and specific level of confidence. It is
mostly to used to estimate the lower bound of the portfolio returns given
its realised returns.

Assume that the realised returns are normally distributed. VaR95 expects
only 5% of observations breaches the threshold deduced by the portfolio
volatility forecast

$$
VaR95_t = Q_{0.05} \sigma_t
$$

where $Q_{0.05}$ is the 5% quantile of the standard normal distribution
and $\sigma_t$ is the volatility forecast of the portfolio deduced by
the risk model, with information up to time $t$.

The rolling-window VaR breach statistics is then defined as

$$
C_T(t) = \frac{1}{T} \sum_{\tau=t-T+1}^{t} 1[r_{\tau} < VaR95_{\tau}]
$$

where $1[x]$ is the indicator function whether x is satisfied, and
$r_t$ is the observed return in the interval of $[t, t+1]$

The expected rolling-window VaR breach statistics should be around
5% (or 1% for VaR99), while the risk model

- overestimates the risk if it is below the expected percentage, and
- underestimates the risk otherwise

## Usage

Assume that you have market returns `returns` in a `DataFrame`, portfolio
weights `weights` in a `DataFrame` (it could be equally weighted, or market
cap weighted), and rolling risk models, which can be a `RollingRiskModel`
object, or just a dictionary of covariances. To compute the bias statistic
in a 21-day rolling window, you can use the following code snippet.

```
from fpm_risk_model.accuracy.value_at_risk import (
  compute_value_at_risk_breach_statistics
)

compute_value_at_risk_breach_statistics(
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
compute_value_at_risk_breach_statistics(
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
.. automodule:: fpm_risk_model.accuracy.value_at_risk
  :members:
```
