from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression


class ModelPipeline:
    def __init__(
        self,
        predictor: BaseEstimator,
        input_scaler: TransformerMixin,
        output_scaler: TransformerMixin,
    ):
        self._predictor = predictor
        self._input_scaler = input_scaler
        self._output_scaler = output_scaler

    def fit(self, X_train, y_train):
        if self._input_scaler:
            X_train = self._input_scaler.fit_transform(X_train)
        if self._output_scaler:
            y_train = self._output_scaler.fit_transform(y_train.reshape(-1, 1)).reshape(
                -1
            )
        self._predictor.fit(X_train, y_train)

    def __call__(self, X):
        if self._input_scaler:
            X = self._input_scaler.transform(X)
        X = self._predictor.predict(X)
        if self._output_scaler:
            X = self._output_scaler.inverse_transform(X.reshape(-1, 1)).reshape(-1)
        return X

    def predict(self, X):
        return self.__call__(X)

    def transform(self, X):
        return self.__call__(X)
