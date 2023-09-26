from fpm_risk_model.engine import LinAlgEngine, NumpyEngine, backend, use_backend


def test_use_backend():
    assert backend() == "numpy"
    with use_backend("tensorflow"):
        assert backend() == "tensorflow"


def test_numpy_engine():
    import numpy

    returns = (numpy.random.rand(100, 20) - 0.5) / 10
    with use_backend("numpy"):
        np = NumpyEngine()
        linalg = LinAlgEngine()
        returns = np.array(returns)
        mu = np.mean(returns, axis=0)
        demean = returns - mu
        cov = demean.T @ demean
        invcov = linalg.inv(cov)
        assert isinstance(invcov, np.ndarray)
