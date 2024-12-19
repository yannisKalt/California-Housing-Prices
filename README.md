# California Housing Prices

## Intro
The core principle of this work is to design *modular* and *dynamic* end-to-end pipelines. The pipeline is divided into two main modules:

- Dataset preprocessing
- Model training and validation
- Model serving

Each of these modules is developped independently by implementing a specific API.
The implementation of each module is defined by a config file. For instance [linear_regression.yaml](configs/model/linear_regression.yaml) constructs a linear regressor while [random_forest_regression.yaml](configs/model/random_forest_regression.yaml) specifies the construction of a random forest. Both of these *predictors* are injected to a model-pipeline [API](src/core/modelling/model_pipeline.py).

Similarly there are config files which implement the end-to-end pipeline.  [all_actions.yaml](configs/actions/all_actions.yaml) specifies which actions in the pipeline will be performed.

Finally, [hyrun.py](src/hyrun.py) runs the whole pipeline specified by the [configuration](configs/config.yaml). Notice that with a single argument we can swap datasets, preprocessing steps, model training and/or serving.


## Setup
Setup a venv install requirements
```
# Windows machine
python -m venv ./.venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Initial playground: Notebooks
[expore_dataset.ipynb](\notebooks/explore_dataset.ipynb) contains the initial playing-around with the dataset. This simple analysis lead to the development of the data preprocessing algorithms and the models.

## Architectural structure
- [src/core](src/core): Contains the code to construct each modules API. <br> For instance [data_preproc](src/core/data_preproc) contains data-preprocessing modules for feature-selection, encoding, imputation etc. 
- [src/actions](src/actions): Each action implements a specific module. Due to dependency-injection each action is flexible on different implementations of each module. For instance [preprocess_data](src/actions/preprocess_data.py) implements a preprocessing pipeline where the **transforms** argument implements the logic unit.
- [src/hyrun](src/hyrun.py): Constructs and runs the pipeline.



### How do I specify the implementation of each module?
As previously mentioned, the main config file specifies both which actions as well as the logic of each module. For instance if you change 

```
- model: random_forest_regression
```
to 
```
- model: linear_regression
```
the pipeline will run with OLS instead of random forest.

### Concluding
While this is a simple implementation, I hope that the main design concepts are clear and insightful. Thanks for your time and consideration :)
