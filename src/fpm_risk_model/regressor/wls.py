from typing import Optional

from numpy import ndarray
from numpy.linalg import pinv


class WLS:
    """
    Weighted least squares problem solver.

    The solver is to run regression with weighted least squares
    objective.
    """

    def __init__(self, executor: str = "closed"):
        """
        Construct

        Parameters
        ----------
        executor : str
          Executor name. Default is "closed", deriving from closed formula.
        """
        self._executor = executor

    def fit(self, X: ndarray, y: ndarray, weights: Optional[ndarray] = None):
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
        if self._executor == "closed":
            return self._close_fit(X=X, y=y, weights=weights)

        raise ValueError(f"Executor {self._executor} is not supported")

    @staticmethod
    def _close_fit(X: ndarray, y: ndarray, weights: Optional[ndarray] = None):
        """
        Fit the coefficients with closed formula.

        coefficients = (X^T @ W @ X)^{-1} @ X^T @ W @ y
        """
        if isinstance(weights, ndarray):
            if len(weights.shape) == 1 and weights.shape[0] == y.shape[0]:
                weights = weights**0.5
                X_t_w = X.T * weights * weights.T
                return pinv(X_t_w @ X) @ X_t_w @ y
            else:
                raise ValueError(
                    f"Dimension of y {y.shape} does not align with weights "
                    f"{weights.shape}"
                )
        else:
            return pinv(X.T @ X) @ X.T @ y
