from typing import Optional

from numpy import diag_indices_from, trace
from pandas import DataFrame, Series

from .risk_model import RiskModel
from .rolling_factor_risk_model import RollingFactorRiskModel


class CovarianceEstimator:
    """
    Covariance estimator.
    """

    def __init__(
        self,
        risk_model: RiskModel,
        shrinkage_method: Optional[str] = None,
        delta: Optional[float] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        risk_model : RiskModel
          Risk model object.
        shrinkage_method : Optional[str]
          Shrinkage method. Options are "constant" and "ledoit_wolf_constant_variance".
        delta : Optional[float]
          Delta, only used in constant shrinkage.
        """
        self._risk_model = risk_model
        self.shrinkage_method = shrinkage_method
        self.delta = delta

    def corr(self) -> DataFrame:
        """
        Correlation
        """
        return self._risk_model.corr()

    def cov(
        self, volatility: Optional[Series] = None, strict: bool = True
    ) -> DataFrame:
        """
        Correlation

        Parameters
        ----------
        volatility : pd.Series
          Volaility series to convert from correlation to covariance. Optional.
        strict : bool
          Indicates to throw exception if volatility series does not align with
          correlation matrix.
        """
        if volatility is None:
            return self._risk_model.cov()

        corr = self._risk_model.corr()
        if strict and set(volatility.index) != set(corr.index):
            raise ValueError(
                "Incorrect volatility series passed. Length of volatility "
                f"is {len(volatility.index)} while that of correlation is "
                f"{len(corr.index)}"
            )
        elif len(volatility.index) < len(corr.index):
            instruments = volatility.index
        else:
            instruments = corr.index

        volatility = volatility.loc[instruments]
        cov = (
            corr.loc[instruments, instruments]
            .mul(volatility, axis=0)
            .mul(volatility, axis=1)
        )

        if self.shrinkage_method == "constant":
            cov = self.constant_shrinkage(cov, self.delta)
        elif self.shrinkage_method is not None:
            raise ValueError(
                f"Cannot recognize shrinkage method {self.shrinkage_method}"
            )

        return cov

    @staticmethod
    def constant_shrinkage(cov: DataFrame, delta: float):
        """
        Constant shrinkage.
        """
        N = len(cov)
        avg_var = trace(cov) / N * delta
        cov *= 1 - delta
        cov.values[diag_indices_from(cov)] += avg_var
        return cov


class RollingCovarianceEstimator:
    """
    Rolling covariance estimator.
    """

    def __init__(
        self,
        rolling_risk_model: RollingFactorRiskModel,
        shrinkage_method: Optional[str] = None,
        delta: Optional[float] = None,
    ):
        """
        Constructor.

        Parameters
        ----------
        rolling_risk_model : RollingFactorRiskModel
          Rolling risk model object.
        shrinkage_method : Optional[str]
          Shrinkage method. Options are "constant" and "ledoit_wolf_constant_variance".
        delta : Optional[float]
          Delta, only used in constant shrinkage.
        """
        self._rolling_risk_model = rolling_risk_model
        self.shrinkage_method = shrinkage_method
        self.delta = delta

    def cov(self, volatility: Optional[DataFrame] = None):
        """
        Correlation

        Parameters
        ----------
        volatility : DataFrame
          Volaility series to convert from correlation to covariance. Optional.
        """
        return {
            date: (
                CovarianceEstimator(
                    risk_model,
                    shrinkage_method=self.shrinkage_method,
                    delta=self.delta,
                ).cov(volatility.loc[date], strict=False)
            )
            for date, risk_model in self._rolling_risk_model.items()
        }
