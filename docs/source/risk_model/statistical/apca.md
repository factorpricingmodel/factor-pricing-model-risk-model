# Asymptotic Principal Component Analysis (APCA)

When PCA is running in rolling window $T$ against a universe
with number of instruments $N$, it requires to satisfy the
condition that $T$ should be much greater than $N$ to produce
quality estimates. To address the shortcoming of PCA, rather
than performing analysis on the $N$ space, the analysis
is performed on the $T$ space.

$$
\hat{Q} = \frac{R R^T}{T} = VDV^T
$$

It is then chosen $V_n^T$, which contains the greatest $n$
eigenvectors, as the factor returns $F$, and run through
the regression on

$$
R = B F + {\Gamma}
$$

## Reference

[Gregory Connor, Robert A. Korajczyk (1988). Risk and return in an equilibrium APT: Application of a new test methodology](https://www.sciencedirect.com/science/article/abs/pii/0304405X88900621?via%3Dihub#preview-section-abstract)

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.statistical.apca
  :members:
```
