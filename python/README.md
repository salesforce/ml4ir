# ml4ir Python Quickstart

## Installation

### Using ml4ir as a library

ml4ir can be installed as a pip package by using the following command

```
pip install ml4ir
```

This will install **[ml4ir-0.0.2](https://pypi.org/project/ml4ir/)** (the current version) from PyPI.

**Requirements**

* python3.7+
* pip3

### Using ml4ir as a toolkit or contributing to ml4ir

##### Firstly, clone ml4ir
```
git clone https://github.com/salesforce/ml4ir
```

You can use and develop on ml4ir either using docker or virtualenv

##### Docker (Recommended)
We have set up a `docker-compose.yml` file for building and using docker containers to train models.

Change the working directory to the python package
```
cd path/to/ml4ir/python/
```

To build the docker image and run unit tests
```
docker-compose up --build
```

To only build the ml4ir docker image without running tests
```
docker-compose build
```

**Requirements**

* [docker](https://www.docker.com/) (18.09+ tested)
* [docker-compose](https://docs.docker.com/compose/)

##### Virtual Environment

Change the working directory to the python package
```
cd path/to/ml4ir/python/
```

Install virtualenv
```
pip3 install virtualenv
```

Create new python3 virtual environment inside your git repo (it's .gitignored, don't worry)
```
python3 -m venv env/.ml4ir_venv3
```

Activate virtualenv
```
source env/.ml4ir_venv3/bin/activate
```

Install all dependencies
```
pip3 install --upgrade setuptools
pip install --upgrade pip
pip3 install -r requirements.txt
```

Set the PYTHONPATH environment variable to point to the python package
```
export PYTHONPATH=$PYTHONPATH:`pwd`
```

**Requirements**
* python3.7+
* pip3

**Note about contributing**

pre-commit-hooks are required, and installed as a requirement for contributing to ml4ir. 
If an error results that they didn't install, execute `pre-commit install` to install git hooks in your .git/ directory.

## Usage

##### ml4ir as a toolkit
The entrypoint into the training or evaluation functionality of ml4ir is through `ml4ir/base/pipeline.py` and for application specific overrides, look at `ml4ir/applications/<eg: ranking>/pipeline.py

Pipelines currently supported:

* `ml4ir/applications/ranking/pipeline.py`

* `ml4ir/applications/classification/pipeline.py`

To run the ml4ir ranking pipeline to train, evaluate and/or test, use
```
docker-compose run ml4ir \
    python3 ml4ir/applications/ranking/pipeline.py \
    <args>
```

An example ranking training predict and evaluate pipeline
```
docker-compose run ml4ir \
	python3 ml4ir/applications/ranking/pipeline.py \
	--data_dir ml4ir/applications/ranking/tests/data/tfrecord \
	--feature_config ml4ir/applications/ranking/tests/data/config/feature_config.yaml \
	--run_id test \
	--data_format tfrecord \
	--execution_mode train_inference_evaluate
```

##### ml4ir as a library

To use ml4ir as a deep learning library to build relevance models, look at the following walkthroughs under `notebooks/`

* **Learning to Rank** : The `PointwiseRankingDemo` notebook walks you through building, training, saving, and the entire life cycle of a `RelevanceModel` from the bottom up. You can also find details regarding the architecture of ml4ir in it.

* **Text Classification** : The `EntityPredictionDemo` notebook walks you through training a model to predict entity type given a user context and query.

Enter the following command to spin up Jupyter notebook on your browser to run the above notebooks
```
jupyter-notebook
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
