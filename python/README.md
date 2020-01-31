# ml4ir: Machine Learning Library for Information Retrieval

## Setup
#### Requirements
* python3.6+
* pip3

#### Virtual Environment
Install virtualenv
```
pip3 install virtualenv
```

Create new python3 virtual environment
```
python3 -m venv env/.ranking_venv3
```

Activate virtualenv
```
cd python/
source env/.ranking_venv3/bin/activate
```

Install all dependencies (carefully)
```
pip3 uninstall setuptools
pip install --upgrade pip
pip3 install -r requirements.txt
```

Note, there are some AWS incompatibilities, gotta fix that:
```
ERROR: botocore 1.14.9 has requirement docutils<0.16,>=0.10, but you'll have docutils 0.16 which is incompatible.
ERROR: awscli 1.17.9 has requirement docutils<0.16,>=0.10, but you'll have docutils 0.16 which is incompatible.
```

Execute `pre-commit install` to install git hooks in your .git/ directory.


Set the PYTHONPATH environment variable
```
export PYTHONPATH=$PYTHONPATH:`pwd`/python
```

## Usage
The entrypoint into the training or evaluation functionality of ml4ir is through `ml4ir/model/pipeline.py`

#### Examples
Using TFRecord
```
python ml4ir/model/pipeline.py \
--data_dir `pwd`/ml4ir/tests/data/tfrecord \
--feature_config `pwd`/ml4ir/tests/data/tfrecord/feature_config.json \
--metrics categorical_accuracy \
--run_id test \
--data_format tfrecord \
--execution_mode train_inference_evaluate
```

Using CSV
```
python ml4ir/model/pipeline.py \
--data_dir `pwd`/ml4ir/tests/data/csv \
--feature_config `pwd`/ml4ir/tests/data/csv/feature_config.json \
--metrics categorical_accuracy \
--run_id test \
--data_format csv \
--execution_mode train_inference_evaluate
```

Running in inference mode using the default serving signature
```
python ml4ir/model/pipeline.py \
--data_dir `pwd`/ml4ir/tests/data/csv \
--feature_config `pwd`/ml4ir/tests/data/csv/feature_config.json \
--metrics categorical_accuracy \
--run_id test \
--data_format csv \
--model_file `pwd`/models/test/final/default \
--execution_mode inference_only
```
## Running Tests
To run all the python based tests under `ml4ir`
```
python3 -m pytest
```

To run specific tests, 
```
python3 -m pytest /path/to/test/module
```

NOTE: Make sure to add the right data and feature config before training models.

## Project Organization

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── ml4ir                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes ml4ir a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.testrun.org


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
