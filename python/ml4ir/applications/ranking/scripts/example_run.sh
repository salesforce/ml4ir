docker-compose run ml4ir \
	python3 ml4ir/applications/ranking/pipeline.py \
	--data_dir ml4ir/applications/ranking/tests/data/tfrecord \
	--feature_config ml4ir/applications/ranking/tests/data/config/feature_config.yaml \
	--run_id test \
	--data_format tfrecord \
	--execution_mode train_inference_evaluate