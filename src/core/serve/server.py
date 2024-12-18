from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model
import pickle
import numpy as np
import os
import json


class Server:
    def __init__(self, model_dir: str, model_tag: str):
        """Simple Rest-API implementation with trivial input-data type validation

        Args:
            model_dir (str): Directory where model data reside.
            model_tag (str): The tag (id) of the model
        """
        self.model = self._load_model(os.path.join(model_dir, model_tag, "model.pkl"))
        self.meta_dict = self._load_meta(
            os.path.join(model_dir, model_tag, "meta.json")
        )

        self.InputFeatures = self._construct_features_model(
            self.meta_dict["input_variables"], "InputFeatures"
        )
        self.OutputResult = self._construct_features_model(
            [self.meta_dict["target_variable"]], "OutputResult", output=True
        )

        self.app = FastAPI()  # Create FastAPI instance
        self._setup_routes()  # Define API routes

    def _construct_features_model(
        self, variables: list, model_name: str, output: bool = False
    ):
        if output:
            # Output model has only one field for the target variable
            fields = {variables[0]: (float, Field(...))}
        else:
            # Input model includes multiple fields
            fields = {var: (float, Field(...)) for var in variables}
        return create_model(model_name, **fields)

    def _load_meta(self, meta_fn: str):
        """
        Load the model input/output featrues
        """
        with open(meta_fn, "r") as f:
            return json.load(f)

    def _load_model(self, model_path: str):
        """
        Load the model from a .pkl file.
        """
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)

    def _setup_routes(self):
        """
        Define API endpoints.
        """

        @self.app.get("/")
        def read_root():
            return {
                "message": f"Model prediction API: Please Insert {list(self.InputFeatures.schema()['properties'].keys())}"
            }

        @self.app.post("/predict", response_model=self.OutputResult)
        def predict(features: self.InputFeatures):
            """
            Predict target variable based on input features.
            """
            # Extract input feature values as a numpy array
            input_data = np.array(
                [[getattr(features, var) for var in self.meta_dict["input_variables"]]]
            )
            # Make prediction using the model
            prediction = self.model.predict(input_data)[0]
            return {self.meta_dict["target_variable"]: prediction}
