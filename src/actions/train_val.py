import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.core.modelling.model_pipeline import ModelPipeline
from typing import Dict, Callable
import mlflow
import pickle
import json


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
    input_variables = sorted(data.columns.drop(target_variable))
    model_meta = {
        "input_variables": input_variables,
        "target_variable": target_variable,
    }
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

    # Save Model
    storage_dir = os.path.join(log_dir, model_tag)
    os.makedirs(storage_dir, exist_ok=True)
    with open(os.path.join(storage_dir, "model.pkl"), "wb") as fn:
        pickle.dump(model_pipeline, fn)
    # Store model input variables
    with open(os.path.join(storage_dir, "meta.json"), "w") as fn:
        json.dump(model_meta, fn)
