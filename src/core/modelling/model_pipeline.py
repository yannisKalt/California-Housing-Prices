from sklearn.base import BaseEstimator


class ModelPipeline:
    def __init__(self, predictor: BaseEstimator, input_scaler, output_scaler):
        self._predictor = predictor

    def fit():
        pass

    def transform():
        pass
