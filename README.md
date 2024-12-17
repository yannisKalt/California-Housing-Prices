# Pfizer_Assessment

## Intro
The core principle of this work is to design *modular* and *dynamic* end-to-end pipelines. The pipeline is divided into two main modules:
- Data Preproc
- Train/val
- Deployment

The logic of each module is defined by a config file. For instance [linear_regression.yaml](configs/model/linear_regression.yaml) constructs a linear regressor while [random_forest_regression.yaml](configs/model/random_forest_regression.yaml) specifies the construction of a random forest. 

Similarly there are config files which implement the pipeline. For instance [all_actions.yaml](configs/actions/all_actions.yaml) specifies which actions in the pipeline will be performed.

Finally, [hyrun.py](src/hyrun.py) runs the whole pipeline specified by the above [configuration](configs/config.yaml). Notice that with a single argument we can swap datasets, preprocessing steps, model training and/or deployment.

For the sake of simplicity and lack of time, the deployment is performed via another script. We'll get into more depth of the architecture later on in this document.

## Setup
Setup a venv i.e for a windows machine and install requirements
```
python -m venv ./.venv
.venv/Scripts/activate
pip install -r requirements.txt
```

## Initial playground: Notebooks
[expore_dataset.ipynb](\notebooks/explore_dataset.ipynb) contains the initial playing-around with the dataset. This toy-analysis lead to the development of the data preproc algos and the selected models.

## Architectural structure
- [src/core](src/core): Contains the code to construct each modules API. <br> For instance [data_preproc](src/core/data_preproc) contains data-preprocessing modules for feature-selection, encoding, imputation etc. 
- [src/actions](src/actions): Each action implements a specific module. Due to dependency-injection each action is flexible on different implementations of each module. For instance [preprocess_data](src/actions/preprocess_data.py) implements a preprocessing pipeline where the **transforms** argument implements the logic unit.


### What happens when i run hyrun.py?
Look at the main config file:
```
defaults:
  - _self_
  - paths: default
  - transforms: default
  - actions: all_actions
  - dataset: housing
  - model: random_forest_regression 
  - metrics: regression
seed: 0
```
[All actions](configs/actions/all_actions.yaml)  will be performed. If you comment out an action, it will be skipped. If you append another action (not implemented) it will be added to the pipeline.

### How do I specify the logic of each module?
The main config file specifies both which actions as well as the logic of each module. For instance if you change 


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