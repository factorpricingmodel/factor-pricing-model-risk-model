from numpy import array, ndarray, sum

from .wls import WLS


class EWLS(WLS):
    """
    Exponential weighted least squares regression.

    The weighting scheme is

    .. math::
        w_t = \\frac{2^{-\\frac{T-t){\\lambda}}}}{\\bar{w}}

    where lambda is the half-life parameter.
    """

    def __init__(self, half_life: float, **kwargs):
        """
        Constructor.

        Parameters
        ----------
        half_life : float
            Half-life of the weighting decay.
        """
        super().__init__(**kwargs)
        self._half_life = half_life

    def fit(self, X: ndarray, y: ndarray):
        """
        Fit the coefficients by the data X and y.

        Parameters
        ----------
        X: ndarray
          Training data.
        y: ndarray
          Target values.
        weights: Optional[ndarray]
          Weightings in regressiond data. The dimension should be
          same as y.
        """
        T = y.shape[0]
        weights = array([2 ** (-(T - 1 - t) / self._half_life) for t in range(0, T)])
        weights /= sum(weights)
        return super().fit(X=X, y=y, weights=weights)
