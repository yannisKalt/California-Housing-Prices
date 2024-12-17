import os
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import StandardScaler


def train_model(
    data_dir,
    data_fn,
    target_variable,
    predictor: BaseEstimator,
    model_tag: str,
    input_scaler: BaseEstimator,
    output_scaler: BaseEstimator,
    tag: str,
    random_state: int = 0,
    test_size: float = 0.2,
):
    # Load Model
    data: pd.DataFrame = pd.read_csv(os.path.join(data_dir, data_fn))
    X = data.drop(columns=target_variable).values
    y = data[target_variable].values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
