import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.modelling.model_pipeline import ModelPipeline
from typing import Dict, Callable
import mlflow
import pickle


def train_val(
    data_dir,
    data_fn,
    log_dir,
    target_variable,
    model_pipeline: ModelPipeline,
    model_tag: str,
    metrics: Dict[str, Callable],
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
    model_pipeline.fit(X_train, y_train)
    y_pred = model_pipeline.predict(X_test)

    # Log Metrics
    with mlflow.start_run(run_name=model_tag):
        mlflow.log_metrics(
            {name: f(y_pred=y_pred, y_true=y_test) for name, f in metrics.items()}
        )
    # Store Model
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, model_tag + ".pkl"), "wb") as fn:
        pickle.dump(model_pipeline, fn)
