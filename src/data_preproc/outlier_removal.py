import pandas as pd
from typing import Optional, Iterable


def iqr(
    df: pd.DataFrame,
    columns: Optional[Iterable[str]] = None,
    q_low: Optional[float] = 0.25,
    q_high: Optional[float] = 0.75,
    scale: Optional[float] = 1.5,
):
    """Perform interquantile range outlier removal

    Args:
        df (pd.DataFrame): The original (encoded) dataframe
        columns (Optional[Iterable[str]], optional): The columns which will be considered for outlier detection
                                                     If None, all columns are considered. Defaults to None.
        q_low (Optional[float], optional): Lower-bound quantile. Defaults to 0.25.
        q_high (Optional[float], optional): Upper-bound quantile. Defaults to 0.75.
        scale (Optional[float], optional): The scaling of the range. Defaults to 1.5.

    Returns:
        pd.DataFrame: The dataframe with removed outlier entries.
    """
    # filter which columns will be subjected to outlier removal
    columns = columns if columns is not None else df.columns
    df_sub = df[columns]
    q_low_value = df_sub.quantile(q_low)
    q_high_value = df_sub.quantile(q_high)
    quantile_range = q_high_value - q_low_value

    lower_bound = q_low_value - quantile_range * scale
    upper_bound = q_high_value + quantile_range * scale
    df = df.loc[
        (df_sub >= lower_bound).all(axis=1) & (df_sub <= upper_bound).all(axis=1)
    ].reset_index(drop=True)
    return df
