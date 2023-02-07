import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models as kmodels

from ml4ir.applications.classification.pipeline import ClassificationPipeline
from ml4ir.applications.classification.tests.test_base import ClassificationTestBase
from ml4ir.base.config.keys import ServingSignatureKey
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.features.preprocessing import split_and_pad_string
from ml4ir.base.model.relevance_model import RelevanceModel


class ClassificationServingTest(ClassificationTestBase):
    """Assess model serving."""

    def test_serve_default_pipeline(self):
        """
        Train a simple model and test serving flow by loading the SavedModel
        """
        # Test model training on TFRecord Example data
        self.set_seeds()
        classification_pipeline: ClassificationPipeline = ClassificationPipeline(
            args=self.get_overridden_args()
        )

        parsed_relevance_dataset: RelevanceDataset = classification_pipeline.get_relevance_dataset()
        raw_relevance_dataset: RelevanceDataset = classification_pipeline.get_relevance_dataset(
            parse_tfrecord=False
        )
        classification_model: RelevanceModel = classification_pipeline.get_relevance_model()

        classification_model.fit(
            dataset=parsed_relevance_dataset, num_epochs=1, models_dir=self.output_dir
        )

        preprocessing_keys_to_fns = {"split_and_pad_string": split_and_pad_string}

        classification_model.save(
            models_dir=self.args.models_dir,
            preprocessing_keys_to_fns=preprocessing_keys_to_fns,
            required_fields_only=True,
        )

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
        sequence_example_protos = next(iter(raw_relevance_dataset.test))
        parsed_sequence_examples = next(iter(parsed_relevance_dataset.test))[0]
        parsed_dataset_batch = parsed_relevance_dataset.test.take(1)

        # Use the loaded serving signatures for inference
        model_predictions = classification_model.predict(parsed_dataset_batch)[
            self.args.output_name
        ].values
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

        # Compare the scores from the different versions of the model
        assert np.isclose(model_predictions[0], default_signature_predictions[0], rtol=0.01,).all()
        assert np.isclose(
            model_predictions[0], tfrecord_signature_predictions[0], rtol=0.01,
        ).all()
