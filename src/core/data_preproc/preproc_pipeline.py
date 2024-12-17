from typing import Callable, List
from collections import defaultdict
import pandas as pd


class DataPreproc:
    def __init__(
        self,
        transforms: List[Callable],
        store_intermediate: bool = False,
    ):
        """Implements a simple transformation pipeline in order to preprocess the dataset

        Args:
            transforms (List[Callable]): A list of sequential transformations.
            store_intermediate (bool, optional): Boolean flag in order to save the intermediate
                                                 data representations. Defaults to False.
        """
        self._store_intermediate = store_intermediate
        self._transforms = transforms
        self._intermediate_data_transforms = defaultdict(int)

    def __call__(self, df: pd.DataFrame):
        for idx, trans in enumerate(self._transforms):
            if trans is not None:
                df = trans(df)
            if self._store_intermediate:
                self._intermediate_data_transforms[idx] = df
        return df

    def get_intermediate_transformations(self):
        return self._intermediate_data_transforms
