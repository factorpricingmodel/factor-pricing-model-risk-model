# Universe

Universe is a set of instruments in portfolio for each period, and the set
of instruments changes along the way.

Generally, there are two types of universe

1. Estimation universe
2. Model universe

Estimation universe is used for modeling the factor risks, and normally
selects the liquid and representative instruments in the entire universe.
For example, in U.S. stock market, an estimation universe can be selected
by the top 40% free-float market capitalisation equities with actively
trading liquidity.

In the meantime, the model universe are selected in a more relaxed manner.
It can be all the stocks in the region sometimes, or a list of instruments
traded in your portfolio. The risk model of model universe is usually
transformed from that generated from estimation universe. In our library,
a risk model of estimation universe can be always transformed with method
`transform` and the returns of the model universe

```
estimation_risk_model = PCA(n_components=5)
estimation_risk_model.fit(X=estimation_returns)

model_risk_model = estimation_risk_model.transform(X=model_returns)
```

In our benchmark metrics, model universes of most instruments in the region
are always used to examine the model forecasting accuracy.

The instrument type is an important selection criteria. For example,
composite assets, like ETF and trusts, are highly correlated with a
combination of subsets of universe. Secondary listing stocks have little
representation in the region but more exposed to the foreign exchange
movement. So, it is important to review the instrument list in the universe
about their constitution and characteristics.

Finally, we have a separate library,
[factor-pricing-model-universe](https://github.com/factorpricingmodel/factor-pricing-model-universe)
to provide functions and pipelines to generate universes. Please refer
to its
[documentation](https://factor-pricing-model-universe.readthedocs.io/en/latest/)
for further details.
