/bin/bash tools/run_docker.sh ml4ir \
	python3 ml4ir/applications/classification/pipeline.py \
	--data_dir ml4ir/applications/classification/tests/data/tfrecord \
	--feature_config ml4ir/applications/classification/tests/data/configs/feature_config.yaml \
	--model_config ml4ir/applications/classification/tests/data/configs/model_config.yaml \
	--batch_size 32 \
	--run_id test \
	--data_format tfrecord \
	--execution_mode train_inference_evaluate
