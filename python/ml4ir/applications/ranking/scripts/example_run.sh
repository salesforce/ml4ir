/bin/bash tools/run_docker.sh ml4ir \
	python3 ml4ir/applications/ranking/model/pipeline.py \
	--data_dir ml4ir/tests/data/tfrecord \
	--feature_config ml4ir/tests/data/tfrecord/feature_config.yaml \
	--run_id test \
	--data_format tfrecord \
	--execution_mode train_inference_evaluate