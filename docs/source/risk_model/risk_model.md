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

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.risk_model
  :members:
```
