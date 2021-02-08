#!/bin/bash

# Script to train a model on a classification dataset in the python project

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}/../../../python

CLASSIFICATION_MODEL=ml4ir/applications/classification

if [ -z "$2" ]
  then
    export PYTHONPATH=.
    EXECUTOR="python"
else
  EXECUTOR="docker-compose run ml4ir python"
fi

$EXECUTOR ${CLASSIFICATION_MODEL}/pipeline.py \
    --data_dir ${CLASSIFICATION_MODEL}/tests/data/csv/  \
    --feature_config ${CLASSIFICATION_MODEL}/tests/data/configs/feature_config.yaml \
    --model_config ${CLASSIFICATION_MODEL}/tests/data/configs/model_config.yaml \
    --run_id ${1} \
    --data_format csv \
    --execution_mode train_inference_evaluate \
    --batch_size 32

cd -
