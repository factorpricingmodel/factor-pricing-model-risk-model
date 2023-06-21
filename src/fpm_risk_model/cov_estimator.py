from typing import Optional

from pandas import DataFrame, Series

from .risk_model import RiskModel
from .rolling_factor_risk_model import RollingFactorRiskModel


class CovarianceEstimator:
    """
    Covariance estimator.
    """

    def __init__(self, risk_model: RiskModel):
        """
        Constructor.

        Parameters
        ----------
        risk_model : RiskModel
          Risk model object.
        """
        self._risk_model = risk_model

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
        return cov


class RollingCovarianceEstimator:
    """
    Rolling covariance estimator.
    """

    def __init__(self, rolling_risk_model: RollingFactorRiskModel):
        """
        Constructor.

        Parameters
        ----------
        rolling_risk_model : RollingFactorRiskModel
          Rolling risk model object.
        """
        self._rolling_risk_model = rolling_risk_model

    def cov(self, volatility: Optional[DataFrame] = None):
        """
        Correlation

        Parameters
        ----------
        volatility : DataFrame
          Volaility series to convert from correlation to covariance. Optional.
        """
        return {
            date: CovarianceEstimator(risk_model).cov(
                volatility.loc[date], strict=False
            )
            for date, risk_model in self._rolling_risk_model.items()
        }
