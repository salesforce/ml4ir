/bin/bash tools/run_docker.sh ml4ir \
	python3 ml4ir/applications/classification/pipeline.py \
	--data_dir ml4ir/applications/classification/tests/data/tfrecord \
	--feature_config ml4ir/applications/classification/tests/data/config/feature_config.yaml \
	--model_config ml4ir/applications/classification/tests/data/config/model_config.yaml \
	--run_id test \
	--data_format tfrecord \
	--execution_mode train_inference_evaluate