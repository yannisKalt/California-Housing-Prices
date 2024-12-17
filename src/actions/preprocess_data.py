from src.core.data_preproc.preproc_pipeline import DataPreproc
import pandas as pd
import os


def preprocess_data(data_dir, data_fn, output_fn, transforms, store_intermediate=False):
    # Read data
    df = pd.read_csv(os.path.join(data_dir, data_fn))
    preproc_pipeline = DataPreproc(
        transforms=transforms.values(), store_intermediate=store_intermediate
    )
    # Transform data
    transformed_df = preproc_pipeline(df)
    transformed_df.to_csv(os.path.join(data_dir, output_fn), index=None)
    if store_intermediate:
        for (
            i,
            intermediate_data_transform,
        ) in preproc_pipeline.get_intermediate_transformations().items():
            fn = f"intermediate_{i + 1}" + output_fn
            intermediate_data_transform.to_csv(os.path.join(data_dir, fn), index=None)
