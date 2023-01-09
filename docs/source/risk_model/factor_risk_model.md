# Factor risk model

A factor risk model is a statistical model that is used to quantify the risk of an investment or portfolio. It takes into account one or more risk factors, or variables that are known to affect the risk of the investment. The model aims to identify the factors that contribute to risk and to measure their impact on the investment's return.

## Description

A factor risk model describes the instrument / portfolio returns $R$ by factor
exposures $B$, factor returns $F$ and residual returns (sometimes named as
idiosyncratic returns) $U$.

$$
R = BF + U
$$

Therefore, a factor risk model object contains the following attributes

- Factor returns: The timeseries of factor returns, in dimension of (T, n)

- Factor covariance: The covariance of the factors derived from the factor
  returns, in dimension of (n, n)

- Factor exposures: The exposure of the instrument / portfolio on each factor,
  in dimension of (n, N)

- Residual returns: The residual returns specific for each instrument /
  portfolio, in dimension of (T, N)

where T, N and n are the number of timeframes, instruments and factors.

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.factor_risk_model
  :members:
```
