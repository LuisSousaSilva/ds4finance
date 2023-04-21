import numpy as np
import pandas as pd

def compute_std_dev(dataframe, freq='auto'):
    """
    Compute the annualized standard deviation of a given DataFrame of prices.

    Parameters
    ----------
    dataframe : pandas.DataFrame
        DataFrame containing price data. Index should be of datetime type.
    freq : str, optional
        Frequency of the data.
        Possible values: 'daily', 'monthly', 'quarterly', 'auto'.
        Default is 'auto', which calculates the scaling factor based on the average
        frequency in the data.

    Returns
    -------
    pandas.Series
        A pandas Series containing the annualized standard deviation
        for each column in the input DataFrame.
    """

    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame")

    # Define a dictionary to map frequency to the appropriate scaling factor
    scaling_factors = {
        'daily': np.sqrt(252),
        'monthly': np.sqrt(12),
        'quarterly': np.sqrt(4)
    }

    # If freq is 'auto', calculate the scaling factor based on the average frequency in
    #  the data
    if freq == 'auto':
        avg_frequency = dataframe.index.to_series().diff().mean()
        sessions_per_year = (365.25 / avg_frequency.days)
        scaling_factor = np.sqrt(sessions_per_year)
    elif freq in scaling_factors:
        scaling_factor = scaling_factors[freq]
    else:
        raise ValueError(
            "Invalid frequency. Allowed values: 'daily', 'monthly', 'quarterly', 'auto'"
        )

    # Compute the percentage change in the price data
    pct_change = dataframe.pct_change()

    # Calculate the standard deviation, scale it according to the scaling factor,
    # and convert to percentage
    annualized_std_dev = pct_change.std().mul(scaling_factor).mul(100)

    return annualized_std_dev