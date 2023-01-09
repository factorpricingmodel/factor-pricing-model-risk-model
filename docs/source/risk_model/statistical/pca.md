# PCA

Assume the historical instrument returns of the estimation universe is
represented by a T x N matrix R. With singular value decomposition (SDV),
the covariance matrix $\hat{Q}$ is decomposed by its eigenvectors and
eigenvalues.

$$
\hat{Q} = \frac{R^T R}{T} = VDV^T
$$

where V is a matrix of eigenvalues (each column is an eigenvector) and
D is a diagonal matrix with eigenvalues $\lambda_i$ in the decreasing
order on the diagonal.

The factor exposure matrix $B$ is taken to be $V_nD^{\frac{1}{2}}_n$, where
n is the number of largest eigenvalues selected, in a dimension of
$(n, N)$.

Factor $F$ (in dimension $(T, n)$) and residual returns ${\Gamma}$ (in
dimension $(T, N)$) can then be computed by either ordinary or
weighted least-squares

$$
& F = (B^T W B)^{-1} B^T W R \\
& {\Gamma} = R - B F
$$

where $W$ is the weight matrix in regression, e.g. an identity matrix in ordinary
weighted least-squares.

## Module

```{eval-rst}
.. automodule:: fpm_risk_model.statistical.pca
  :members:
```
