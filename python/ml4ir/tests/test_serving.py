from ml4ir.tests.test_base import RankingTestBase
from ml4ir.data.ranking_dataset import RankingDataset
from ml4ir.model.ranking_model import RankingModel
from ml4ir.config.features import parse_config, Features
import os
import numpy as np
from tensorflow.keras import models as kmodels
from ml4ir.config.keys import ServingSignatureKey
import tensorflow as tf


class RankingModelTest(RankingTestBase):
    def test_model_serving(self):
        """
        Train a simple model and test serving flow by loading the SavedModel
        """

        # Test model training on TFRecord SequenceExample data
        data_dir = os.path.join(self.root_data_dir, "tfrecord")
        feature_config_path = os.path.join(
            self.root_data_dir, "tfrecord", self.feature_config_fname
        )

        feature_config: Features = parse_config(feature_config_path)

        self.args.metrics = ["categorical_accuracy"]

        def get_dataset(parse_tfrecord):
            return RankingDataset(
                data_dir=data_dir,
                data_format="tfrecord",
                features=feature_config,
                max_num_records=self.args.max_num_records,
                loss_key=self.args.loss,
                scoring_key=self.args.scoring,
                batch_size=self.args.batch_size,
                train_pcent_split=self.args.train_pcent_split,
                val_pcent_split=self.args.val_pcent_split,
                test_pcent_split=self.args.test_pcent_split,
                parse_tfrecord=parse_tfrecord,
                logger=self.logger,
            )

        # Get raw TFRecord dataset
        raw_dataset = get_dataset(parse_tfrecord=False)

        # Parse the raw TFRecord dataset
        parsed_dataset = get_dataset(parse_tfrecord=True)

        model = RankingModel(
            architecture_key=self.args.architecture,
            loss_key=self.args.loss,
            scoring_key=self.args.scoring,
            metrics_keys=self.args.metrics,
            optimizer_key=self.args.optimizer,
            features=feature_config,
            max_num_records=self.args.max_num_records,
            model_file=self.args.model_file,
            learning_rate=self.args.learning_rate,
            learning_rate_decay=self.args.learning_rate_decay,
            compute_intermediate_stats=self.args.compute_intermediate_stats,
            logger=self.logger,
        )

        model.fit(dataset=parsed_dataset, num_epochs=1, models_dir=self.output_dir)

        model.save(models_dir=self.args.models_dir)

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
        parsed_sequence_examples = {
            k: tf.cast(v, tf.float32) for k, v in next(iter(parsed_dataset.test))[0].items()
        }
        parsed_dataset_batch = parsed_dataset.test.take(1)

        # Use the loaded serving signatures for inference
        model_predictions = model.predict(parsed_dataset_batch)
        default_signature_predictions = default_signature(**parsed_sequence_examples)[
            "ranking_scores"
        ]
        tfrecord_signature_predictions = tfrecord_signature(
            sequence_example_protos=sequence_example_protos
        )["ranking_scores"]

        # Get mask for padded values
        mask = parsed_sequence_examples["mask"]

        # Compare the scores from the different versions of the model
        assert np.isclose(
            np.where(np.equal(mask, 0), 0.0, model_predictions),
            np.where(np.equal(mask, 0), 0.0, default_signature_predictions),
            rtol=0.01,
        ).all()

        assert np.isclose(
            np.where(np.equal(mask, 0), 0.0, model_predictions),
            np.where(np.equal(mask, 0), 0.0, tfrecord_signature_predictions),
            rtol=0.01,
        ).all()

        assert np.isclose(
            np.where(np.equal(mask, 0), 0.0, default_signature_predictions),
            np.where(np.equal(mask, 0), 0.0, tfrecord_signature_predictions),
            rtol=0.01,
        ).all()
