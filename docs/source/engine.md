# Backend Engine

Currently, the default engine (and the array type) is NumPy. Users can switch to various of supported
backend engines for computational acceleration. Currently the following engines are supported in
CPU and GPU runtime.

- [NumPy (Default)](https://numpy.org/)

- [JAX](https://jax.readthedocs.io/en/latest/index.html#)

- [TensorFlow](https://www.tensorflow.org/guide/tf_numpy)

- [PyTorch](https://pytorch.org/)

- [Dask](https://docs.dask.org/en/stable/)

The backend engine can be switched in the following ways.

## Local

Users can use context manager `use_backend` in a `with` statement to temporarily
switch the NumPy engine.

In the following example, to fit a PCA model, a NumPy array of daily returns `daily_returns`
is passed into `fit` method. Use the function `use_backend` to switch the engine temporarily
from NumPy to Tensorflow

```
from fpm_risk_model.engine import use_backend

with use_backend("tensorflow"):
    model = PCA(n_components=10, speedup=False)
    model.fit(daily_returns)
```

Looking into the type of the fitted factor returns, you will see `Tensor` type rather than
NumPy `array` type.

```
print(model.factor_returns.__class__.__name__)    # Tensor type
```

## Global

In the meantime, users can switch the backend engine in global with function `set_backend`.

For example, the following code switches the library backend to Tensorflow in the whole session.

```
from fpm_risk_model.engine import set_backend

set_backend("tensorflow")
```

## Environment variable

Set the environment variable `FPM_BACKEND_ENGINE` before starting the process.

For example, to start the IPython with setting the default backend engine to `torch`, launch the
terminal as below

```
FPM_BACKEND_ENGINE=torch ipython
```

## Reference

For further details, please find the following notebook [example](https://colab.research.google.com/github/factorpricingmodel/factor-pricing-model-risk-model/blob/main/examples/notebook/numpy_backend_engine.ipynb).
