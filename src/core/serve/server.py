from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model
import pickle
import numpy as np
import os
import json


class Server:
    def __init__(self, model_dir, model_tag):
        """
        Initialize the Server with a model loaded from the given path.
        """
        self.model = self.load_model(os.path.join(model_dir, model_tag, "model.pkl"))
        self.meta_dict = self.load_meta(os.path.join(model_dir, model_tag, "meta.json"))

        self.InputFeatures = self.construct_features_model(
            self.meta_dict["input_variables"], "InputFeatures"
        )
        self.OutputResult = self.construct_features_model(
            [self.meta_dict["target_variable"]], "OutputResult", output=True
        )

        self.app = FastAPI()  # Create FastAPI instance
        self.setup_routes()  # Define API routes

    def construct_features_model(
        self, variables: list, model_name: str, output: bool = False
    ):
        """
        Dynamically create a Pydantic model for input or output features.
        """
        if output:
            # Output model has only one field for the target variable
            fields = {variables[0]: (float, Field(...))}
        else:
            # Input model includes multiple fields
            fields = {var: (float, Field(...)) for var in variables}
        return create_model(model_name, **fields)

    def load_meta(self, meta_fn: str):
        """
        Load the model input/output featrues
        """
        with open(meta_fn, "r") as f:
            return json.load(f)

    def load_model(self, model_path: str):
        """
        Load the model from a .pkl file.
        """
        with open(model_path, "rb") as model_file:
            return pickle.load(model_file)

    def setup_routes(self):
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
