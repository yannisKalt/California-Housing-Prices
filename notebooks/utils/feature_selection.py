from pandas import DataFrame
from typing import List
from scipy.stats import pearsonr, f_oneway
import numpy as np
from sklearn.feature_selection import mutual_info_regression as mi


def df_numerical_corr(df: DataFrame, input_variables: List[str], target_variable: str):
    """Computes the pearson-r correlation between multiple numerical input-variables and
       a single numerical target-variable in a dataframe.

    Args:
        df (DataFrame): A dataframe containing all variables (columns).
        input_variables (List[str]): Column names which are consider input-variables.
        target_variable (str): Column name of the target-variable.

    Returns:
        Dataframe: A dataframe with the corresponding correlations (r) and p-values (p) as columns.
                   Each index (row) corresponds to a input-variable name.

    """
    results = DataFrame(index=input_variables, columns=["r", "p"])
    for var in input_variables:
        x, y = df[[var, target_variable]].dropna(axis=0, how="any").values.T
        r, p = pearsonr(x=x, y=y)
        results.loc[var] = (r, p)
    return results.sort_values(by="r")


def df_nominal_eta_corr(
    df: DataFrame, input_variables: List[str], target_variable: str
):
    """Performs ETA correlation coefficient between a list of categorical input-variables and a numerical target-variable.

    Args:
        df (DataFrame): A dataframe containing all variables (columns).
        input_variables (List[str]): Column names which are consider input-variables.
        target_variable (str): Column name of the target-variable.
    Returns:
         Dataframe: A dataframe with the corresponding η^2 values (eta_squared), annova f_score (f_score) and p-value (p) columns.
                   Each index (row) corresponds to a input-variable name.
    """
    results = DataFrame(index=input_variables, columns=["eta_squared", "f_score", "p"])
    for var in input_variables:
        x, y = df[[var, target_variable]].dropna(axis=0, how="any").values.T
        grouped_target_values = [y[x == cat] for cat in np.unique(x)]
        f_score, p_value = f_oneway(*grouped_target_values)
        overall_mean = y.mean()
        ss_total = np.sum((y - overall_mean) ** 2)
        ss_between = sum(
            len(group) * (group.mean() - overall_mean) ** 2
            for group in grouped_target_values
        )
        eta_squared = ss_between / ss_total
        results.loc[var] = (eta_squared, f_score, p_value)
    return results


def df_mutual_info(df: DataFrame, input_variables: List[str], target_variable: str):
    X = df[input_variables].values
    y = df[target_variable].values
    results = DataFrame(columns=["Mutual Info"], index=input_variables)
    results.loc[input_variables, "Mutual Info"] = mi(X, y)
    return results
