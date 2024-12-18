from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional


class ModelPipeline:
    def __init__(
        self,
        predictor: BaseEstimator,
        input_scaler: Optional[TransformerMixin] = None,
        output_scaler: Optional[TransformerMixin] = None,
    ):
        """Model Pipeline API

        Args:
            predictor (BaseEstimator): sklearn based predictor
            input_scaler (Optional[TransformerMixin], optional): Input data scaler. Defaults to None.
            output_scaler (Optional[TransformerMixin], optional): Target variable scaler. Defaults to None.
        """
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
