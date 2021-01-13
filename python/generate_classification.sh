#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd ${DIR}

CLASSIFICATION_MODEL=ml4ir/applications/classification

PYTHONPATH=. python ${CLASSIFICATION_MODEL}/pipeline.py \
    --data_dir ${CLASSIFICATION_MODEL}/tests/data/csv/  \
    --feature_config ${CLASSIFICATION_MODEL}/tests/data/configs/feature_config.yaml \
    --model_config ${CLASSIFICATION_MODEL}/tests/data/configs/model_config.yaml \
    --run_id end_to_end_classif \
    --data_format csv \
    --execution_mode train_inference_evaluate \
    --batch_size 32

cd -
