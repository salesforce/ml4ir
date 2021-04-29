import os
import numpy as np
from tensorflow.keras import models as kmodels
import tensorflow as tf

from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.config.keys import DataFormatKey
from ml4ir.applications.ranking.model.ranking_model import RankingModel
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import ServingSignatureKey


class RankingModelTest(RankingTestBase):
    def test_model_serving(self):
        """
        Train a simple model and test serving flow by loading the SavedModel
        """

        # Test model training on TFRecord SequenceExample data
        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        feature_config: FeatureConfig = self.get_feature_config()

        metrics_keys = ["categorical_accuracy"]

        def get_dataset(parse_tfrecord):
            return RelevanceDataset(
                data_dir=data_dir,
                data_format=DataFormatKey.TFRECORD,
                feature_config=feature_config,
                tfrecord_type=self.args.tfrecord_type,
                max_sequence_size=self.args.max_sequence_size,
                batch_size=self.args.batch_size,
                preprocessing_keys_to_fns={},
                train_pcent_split=self.args.train_pcent_split,
                val_pcent_split=self.args.val_pcent_split,
                test_pcent_split=self.args.test_pcent_split,
                use_part_files=self.args.use_part_files,
                parse_tfrecord=parse_tfrecord,
                file_io=self.file_io,
                logger=self.logger,
            )

        # Get raw TFRecord dataset
        raw_dataset = get_dataset(parse_tfrecord=False)

        # Parse the raw TFRecord dataset
        parsed_dataset = get_dataset(parse_tfrecord=True)

        model: RankingModel = self.get_ranking_model(
            loss_key=self.args.loss_key, feature_config=feature_config, metrics_keys=metrics_keys
        )

        model.fit(dataset=parsed_dataset, num_epochs=1, models_dir=self.output_dir)

        model.save(
            models_dir=self.args.models_dir,
            preprocessing_keys_to_fns={},
            postprocessing_fn=None,
            required_fields_only=not self.args.use_all_fields_at_inference,
            pad_sequence=self.args.pad_sequence_at_inference,
        )

        # Load SavedModel and get the right serving signature
        default_model = kmodels.load_model(
            os.path.join(self.output_dir, "final", "default"), compile=False
        )
        assert ServingSignatureKey.DEFAULT in default_model.signatures
        default_signature = default_model.signatures[ServingSignatureKey.DEFAULT]

        tfrecord_model = kmodels.load_model(
            os.path.join(self.output_dir, "final", "tfrecord"), compile=False
        )
        assert ServingSignatureKey.TFRECORD in tfrecord_model.signatures
        tfrecord_signature = tfrecord_model.signatures[ServingSignatureKey.TFRECORD]

        # Fetch a single batch for testing
        sequence_example_protos = next(iter(raw_dataset.test))
        parsed_sequence_examples = next(iter(parsed_dataset.test))[0]
        parsed_dataset_batch = parsed_dataset.test.take(1)

        # Use the loaded serving signatures for inference
        model_predictions = model.predict(parsed_dataset_batch)[self.args.output_name].values
        default_signature_predictions = default_signature(**parsed_sequence_examples)[
            self.args.output_name
        ]

        # Since we do not pad dummy records in tfrecord serving signature,
        # we can only predict on a single record at a time
        tfrecord_signature_predictions = [
            tfrecord_signature(protos=tf.gather(sequence_example_protos, [i]))[
                self.args.output_name
            ]
            for i in range(self.args.batch_size)
        ]

        def _flatten_records(x):
            """Collapse first two dimensions of a tensor -> [batch_size, max_num_records]"""
            return tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], axis=0))

        def _filter_records(x, mask):
            """
            Filter records that were padded in each query
            Input shape: [batch_size, num_features]
            Output shape: [batch_size, num_features]
            """
            return tf.squeeze(tf.gather_nd(x, tf.where(tf.not_equal(mask, 0))))

        # Get mask for padded values
        mask = _flatten_records(parsed_sequence_examples["mask"])

        # Flatten scores to each record and filter out scores from padded records
        default_signature_predictions = _filter_records(
            _flatten_records(default_signature_predictions), mask
        )
        tfrecord_signature_predictions = tf.squeeze(
            tf.concat(tfrecord_signature_predictions, axis=1)
        )

        # Compare the scores from the different versions of the model
        assert np.isclose(model_predictions, default_signature_predictions, rtol=0.01,).all()

        assert np.isclose(model_predictions, tfrecord_signature_predictions, rtol=0.01,).all()

        assert np.isclose(
            default_signature_predictions, tfrecord_signature_predictions, rtol=0.01,
        ).all()

    def get_feature_config(self):
        feature_config_path = os.path.join(
            self.root_data_dir, "configs", self.feature_config_fname
        )

        feature_config: FeatureConfig = FeatureConfig.get_instance(
            tfrecord_type=self.args.tfrecord_type,
            feature_config_dict=self.file_io.read_yaml(feature_config_path),
            logger=self.logger,
        )

        return feature_config

    def get_tfrecord_signature(self, feature_config: FeatureConfig):
        metrics_keys = ["categorical_accuracy"]
        model: RankingModel = self.get_ranking_model(
            loss_key=self.args.loss_key, feature_config=feature_config, metrics_keys=metrics_keys
        )
        model.save(
            models_dir=self.args.models_dir,
            preprocessing_keys_to_fns={},
            postprocessing_fn=None,
            required_fields_only=not self.args.use_all_fields_at_inference,
            pad_sequence=self.args.pad_sequence_at_inference,
        )

        # Load SavedModel and get the right serving signature
        tfrecord_model = kmodels.load_model(
            os.path.join(self.output_dir, "final", "tfrecord"), compile=False
        )
        assert ServingSignatureKey.TFRECORD in tfrecord_model.signatures

        return tfrecord_model.signatures[ServingSignatureKey.TFRECORD]

    def test_serving_n_records(self):
        """Test serving signature with different number of records"""
        feature_config: FeatureConfig = self.get_feature_config()
        tfrecord_signature = self.get_tfrecord_signature(feature_config)

        for num_records in range(1, 120):
            proto = tf.constant(
                [feature_config.create_dummy_protobuf(num_records=num_records).SerializeToString()]
            )
            try:
                tfrecord_signature(protos=proto)
            except Exception:
                assert False

    def test_serving_required_fields_only(self):
        """Test serving signature with protos with only required fields"""
        feature_config: FeatureConfig = self.get_feature_config()
        tfrecord_signature = self.get_tfrecord_signature(feature_config)

        proto = tf.constant(
            [feature_config.create_dummy_protobuf(required_only=True).SerializeToString()]
        )

        try:
            tfrecord_signature(protos=proto)
        except Exception:
            assert False
