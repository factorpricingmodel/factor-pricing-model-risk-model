# Value at risk (VaR)

Value at risk (VaR) is a risk metric representing the maximum possible
loss for a given holding horizen and specific level of confidence. It is
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

## Reference

Alexander, Carol (2009). Market risk analysis, value at risk models. John Wiley & Sons.

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.accuracy.value_at_risk
  :members:
```
