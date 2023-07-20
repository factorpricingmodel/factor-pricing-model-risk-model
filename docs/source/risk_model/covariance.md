# Risk model covariance

For the factor risk model, the covariance is computed by

$$
Q = B \Sigma B^T + \Delta ^ 2
$$

where $B$ is the factor exposure, $\Sigma$ is the factor
covariance and $\Delta$ is the specific variance (sample
variance of residual returns).

## Unmodified

The unmodified covariances can be returned from the factor model directly by
method `cov`, and same for correlations with `corr`.

For example, covariances and correlation on a single factor model can be
retrieved as follows.

```
from fpm_risk_model import FactorModel
factor_model = FactorModel(...)
cov = factor_model.cov()
corr = factor_model.corr()
```

For rolling factor model, the returns are a dict of covariances with dates
as keys.

## Covariance estimator

Covariance estimators provide advanced features in transforming the
factor exposures and returns into pairwise covariances and correlations.

### Covariance shrinkage

The usual approach to unbiased estimator is a maximum likelihood estimator and
the approach converges to the population variance with Law of large numbers.

However, it is not a good estimator of the eigenvalues of the covariance matrix,
and especially its inverse. Covariance estimation requires a large sample of
data to converge to its true population. Due to curse of dimensionality, the
estimate on a smaller set of observations yields to a noisy sample covariance,
and even its noisier inverse. Also, it is a general phenomenon of a finite
sample data with extreme and possibly noisy returns in financial data.

Covariance shrinkage is a technique to produce a greater signal-to-noise ratio
on covariance estimation. Primarily, covariance shrinkage takes a parameter
$\delta$ to suppress its off-diagonal elements on covariance (or actually its
correlation) matrix.

$$
\hat{Q} = (1 - \delta) * Q + \delta * \mu * I
$$

where $\mu$ is the mean of all diagonal entries (variances) of $Q$.

Covariance shrinkage can be specified in the covariance estimator creation.
For example, to specify a constant shrinkage with delta equal to 0.2,

```
from fpm_risk_model import RollingCovarianceEstimator

estimator = RollingCovarianceEstimator(
  rolling_risk_model,
  shrinkage_method="constant",
  delta=0.2
)
estimator.cov()
```

### Volatility adjustments

Unlike covariance estimation, volatility estimation does not require as much data
and more options, like GARCH, in volatility estimation are available to improve
the short-term forecast performance. So it is a common practice to generate the
covariance with medium / long-term observations and then adjust the variances
(diagonal entries) with estimations from short-term observations.

For example, GARCH volatility estimation is computed as `garch_est`, and the
volatilities can be adjusted with argument `volatility` in `cov` method,

```
estimator.cov(volatility=garch_est)
```

## Reference

[Ledoit, O., & Wolf, M. (2004). Honey, I Shrunk the Sample Covariance Matrix.
The Journal of Portfolio Management, 30(4), 110–119.](https://doi.org/10.3905/jpm.2004.110)

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.cov_estimator
  :members:
```
