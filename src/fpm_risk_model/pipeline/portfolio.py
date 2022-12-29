from pandas import DataFrame


def equal_weighted_portfolio(validity: DataFrame) -> DataFrame:
    """
    Construct an equal weighted portfolio from validity.

    Parameters
    ----------
    validity : DataFrame
        DataFrame containing the universe validity.

    Returns
    -------
    DataFrame
        DataFrame containing the weights of the equal weighted portfolio.
    """
    weights = DataFrame(
        1.0,
        index=validity.index,
        columns=validity.columns,
    )
    weights = weights.where(validity)
    weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)
    return weights
