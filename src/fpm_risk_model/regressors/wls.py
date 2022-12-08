from typing import Union
from numpy.linalg import pinv
from numpy import ndarray, array


class WLS:
    def __init__(self, executor: str = "closed"):
        """
        Construct

        Parameters
        ----------
        executor : str
          Executor name. Default is "closed", deriving from closed formula.
        """
        self._executor = executor

    def fit(self, X: ndarray, y: ndarray, weights: Union[ndarray, float] = 1.0):
        """
        Fit the coefficients by the data X and y.

        Parameters
        ----------
        X: ndarray
          Training data.
        y: ndarray
          Target values.
        weights: Union[ndarray, float]
          Weightings in regressiond data. The dimension should be
          same as y.
        """
        if self._executor == "closed":
            return self._close_fit(X=X, y=y, weights=weights)

        raise ValueError(f"Executor {self._executor} is not supported")

    @staticmethod
    def _close_fit(X: ndarray, y: ndarray, weights: Union[ndarray, float] = 1.0):
        """
        Fit the coefficients with closed formula.

        coefficients = (X^T @ W @ X)^{-1} @ X^T @ W @ y
        """
        return pinv(X.T * weights @ X) @ X.T * weights @ y
