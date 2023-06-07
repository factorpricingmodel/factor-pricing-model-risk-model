# Factor Pricing Model Risk Model

<p align="center">
  <a href="https://github.com/factorpricingmodel/factor-pricing-model-risk-model/actions?query=workflow%3ACI">
    <img src="https://github.com/factorpricingmodel/factor-pricing-model-risk-model/actions/workflows/ci.yml/badge.svg" alt="CI Status" >
  </a>
  <a href="https://factor-pricing-model-risk-model.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/factor-pricing-model-risk-model.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/factorpricingmodel/factor-pricing-model-risk-model">
    <img src="https://img.shields.io/codecov/c/github/factorpricingmodel/factor-pricing-model-risk-model.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAASCAYAAABrXO8xAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAJJSURBVHgBfZLPa1NBEMe/s7tNXoxW1KJQKaUHkXhQvHgW6UHQQ09CBS/6V3hKc/AP8CqCrUcpmop3Cx48eDB4yEECjVQrlZb80CRN8t6OM/teagVxYZi38+Yz853dJbzoMV3MM8cJUcLMSUKIE8AzQ2PieZzFxEJOHMOgMQQ+dUgSAckNXhapU/NMhDSWLs1B24A8sO1xrN4NECkcAC9ASkiIJc6k5TRiUDPhnyMMdhKc+Zx19l6SgyeW76BEONY9exVQMzKExGKwwPsCzza7KGSSWRWEQhyEaDXp6ZHEr416ygbiKYOd7TEWvvcQIeusHYMJGhTwF9y7sGnSwaWyFAiyoxzqW0PM/RjghPxF2pWReAowTEXnDh0xgcLs8l2YQmOrj3N7ByiqEoH0cARs4u78WgAVkoEDIDoOi3AkcLOHU60RIg5wC4ZuTC7FaHKQm8Hq1fQuSOBvX/sodmNJSB5geaF5CPIkUeecdMxieoRO5jz9bheL6/tXjrwCyX/UYBUcjCaWHljx1xiX6z9xEjkYAzbGVnB8pvLmyXm9ep+W8CmsSHQQY77Zx1zboxAV0w7ybMhQmfqdmmw3nEp1I0Z+FGO6M8LZdoyZnuzzBdjISicKRnpxzI9fPb+0oYXsNdyi+d3h9bm9MWYHFtPeIZfLwzmFDKy1ai3p+PDls1Llz4yyFpferxjnyjJDSEy9CaCx5m2cJPerq6Xm34eTrZt3PqxYO1XOwDYZrFlH1fWnpU38Y9HRze3lj0vOujZcXKuuXm3jP+s3KbZVra7y2EAAAAAASUVORK5CYII=" alt="Poetry">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/factor-pricing-model-risk-model/">
    <img src="https://img.shields.io/pypi/v/factor-pricing-model-risk-model.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/factor-pricing-model-risk-model.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/factor-pricing-model-risk-model.svg?style=flat-square" alt="License">
</p>

Package to build risk model for factor pricing model. For further details, please refer
to the [documentation](https://factor-pricing-model-risk-model.readthedocs.io/en/latest/)

## Installation

Install this via pip (or your favourite package manager):

`pip install factor-pricing-model-risk-model`

## Usage

The library contains the pipelines to build the risk model. You can
run the pipelines interactively in Jupyter Notebook.

```python
import fpm_risk_model
```

## Objective

The project provides frameworks to create multi-factor risk
model on an "enterprise-like" level.

The target audiences are researchers, developers and fund
management looking for flexibility in creating risk models.

## Examples

For end-to-end examples, please refer to [examples](https://github.com/factorpricingmodel/factor-pricing-model-risk-model/tree/main/examples) for the below notebooks

- [Cryptocurrency Statistical Risk Model](https://colab.research.google.com/github/factorpricingmodel/factor-pricing-model-risk-model/blob/main/examples/notebook/crypto_statistical_risk_model.ipynb)

## Features

Basically, there are three major features provided in the library

- Factor risk model creation
- Covariance estimator
- Tracking risk model accuracy

## Factor risk model

The factor risk model is created by fitting instrument returns (which
could be weekly, daily, or even higher granularity) and other related
parameters into the model, and its products are factor exposures,
factor returns, factor covariance, and residual returns (idiosyncratic
returns).

For example, to create a simple statistical PCA risk model,

```
from fpm_risk_model.statistics import PCA

risk_model = PCA(n_components=5)
risk_model.fit(X=returns)

# Get fitted factor exposures
risk_model.factor_exposures
```

Then, the risk model can be transformed by the returns of a
larger homogeneous universe.

```
risk_model.transform(y=model_returns)
```

For further details, please refer to the [section](https://factor-pricing-model-risk-model.readthedocs.io/en/latest/risk_model/factor_risk_model.html) in the documentation.

## Covariance estimation

Currently, covariance estimation is supported in factor risk model,
and the estimation depends on the fitted results.

For example, a risk model transformed by model universe returns can
derive the pairwise covariance and correlation for the model universe.

```
risk_model.transform(y=model_returns)

cov = risk_model.cov()
corr = risk_model.corr()
```

The following features will be supported in the near future

- Covariance shrinkage
- Covariance estimation from returns

For further details, please refer to the [section](https://factor-pricing-model-risk-model.readthedocs.io/en/latest/risk_model/covariance.html) in the documentation.

## Tracking risk model accuracy

The library also focuses on the predictability interpretation of the risk
model, and provides a few benchmarks to examine the following metrics

- [Bias](https://factor-pricing-model-risk-model.readthedocs.io/en/latest/accuracy/bias.html)
- [Value at Risk (VaR)](https://factor-pricing-model-risk-model.readthedocs.io/en/latest/accuracy/value_at_risk.html)

For example, to examine the bias statistics of a risk model regarding
an equally weighted portfolio (of which its weights are denoted as `weights`),
pass the instrument observed returns (denoted as `returns`), and either
a rolling risk model (to compute the volatility forecast) or a time series
of volatility forecasts.

```
from fpm_risk_model.accuracy import compute_bias_statistics
compute_bias_statistics(
  X=returns,
  weights=weights,
  window=window
  ...
)
```

## Roadmap

The following major features will be enhanced

- Factor exposures computation from factor returns (Q2 2023)
- Shrinking covariance (Q2 2023)
- Exponential decay weighted least squares regression (Q3 2023)
- Multiple types of running engine, e.g. Tensorflow (Q3 2023)
- Multi-asset class factor model (Q3 2023)
- Fundamental type risk model (Q4 2023)

## Contribution

All levels of contributions are welcomed. Please refer to the [contributing](https://factor-pricing-model-risk-model.readthedocs.io/en/latest/contributing.html)
section for development and release guidelines.
