import pandas as pd
from typing import Iterable, Optional
import numpy as np


def one_hot_encode(
    df: pd.DataFrame,
    columns: Iterable[str],
    prefix: Optional[str] = None,
    prefix_sep: Optional[str] = "_",
):
    """Performs one-hot encoding of a dataframe's categorical variables.

    Args:
        df (pd.DataFrame): The Dataframe containing the data.
        columns (Iterable[str]): The list of categorical variable names.
        prefix (Optional[str]): Prefix added to the new one-hotted columns
        prefix_sep (Optional[str]): Prefix separation added to the new one-hotted columns

    Returns:
        pd.Dataframe: The original dataframe where the categorical variables are replaced
                      with one-hot encoded ones.
    """
    return pd.get_dummies(
        data=df, columns=columns, prefix=prefix, prefix_sep=prefix_sep, dtype=float
    )
