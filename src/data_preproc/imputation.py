import pandas as pd
from typing import Iterable


def stat_imputation(df: pd.DataFrame, columns: Iterable[str], mode: str = "median"):
    return df.fillna(getattr(df[columns], mode)(skipna=True))
