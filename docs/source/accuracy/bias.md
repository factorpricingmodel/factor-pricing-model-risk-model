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

## Reference

Alexander, Carol (2009). Market risk analysis, value at risk models. John Wiley & Sons.

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.accuracy.bias
  :members:
```
