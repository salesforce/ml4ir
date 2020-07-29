import os
import numpy as np
from tensorflow.keras import models as kmodels

from ml4ir.applications.classification.pipeline import ClassificationPipeline
from ml4ir.applications.classification.tests.test_base import ClassificationTestBase
from ml4ir.base.config.keys import ServingSignatureKey
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.model.relevance_model import RelevanceModel


class ClassificationServingTest(ClassificationTestBase):
    """Assess model serving."""

    def test_serve_default_pipeline(self):
        """
        Train a simple model and test serving flow by loading the SavedModel
        """
        # Test model training on TFRecord Example data
        self.set_seeds()
        classification_pipeline: ClassificationPipeline = ClassificationPipeline(args=self.get_overridden_args())

        relevance_dataset: RelevanceDataset = classification_pipeline.get_relevance_dataset()
        classification_model: RelevanceModel = classification_pipeline.get_relevance_model()

        classification_model.fit(dataset=relevance_dataset,
                                 num_epochs=1,
                                 models_dir=self.output_dir
        )

        ###########################################################################################################
        # Model saving manually instead of calling classification_model.save() because currently only             #
        # SequenceExample inputs are supported and we face https://github.com/tensorflow/tensorflow/issues/31686. #
        model_file = os.path.join(self.args.models_dir, "final")                                                  #
        # Save model with default signature                                                                       #
        classification_model.model.save(filepath=os.path.join(model_file, "default"))                             #
        ###########################################################################################################

        default_model = kmodels.load_model(
            os.path.join(self.output_dir, "final", "default"), compile=False
        )
        assert ServingSignatureKey.DEFAULT in default_model.signatures
        default_signature = default_model.signatures[ServingSignatureKey.DEFAULT]

        # Fetch a single batch for testing
        parsed_sequence_examples = next(iter(relevance_dataset.test))[0]
        parsed_dataset_batch = relevance_dataset.test.take(1)

        # Use the loaded serving signatures for inference
        model_predictions = classification_model.predict(parsed_dataset_batch)[self.args.output_name].values
        default_signature_predictions = default_signature(**parsed_sequence_examples)[
            self.args.output_name
        ]

        # Compare the scores from the different versions of the model
        assert np.isclose(model_predictions[0], default_signature_predictions[0], rtol=0.01, ).all()

    def get_tfrecord_signature(self):
        self.set_seeds()
        classification_pipeline: ClassificationPipeline = ClassificationPipeline(args=self.get_overridden_args())

        classification_model: RelevanceModel = classification_pipeline.get_relevance_model()

        classification_model.save(
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

    def get_feature_config(self):
        feature_config: FeatureConfig = FeatureConfig.get_instance(
            tfrecord_type=self.args.tfrecord_type,
            feature_config_dict=self.feature_config,
            logger=self.logger,
        )

        return feature_config

####################################################################################################
# Test below disabled as we currently face: https://github.com/tensorflow/tensorflow/issues/31686. #
####################################################################################################
#    def test_serving_n_records(self):
#        """Test serving signature with different number of records"""
#        feature_config: FeatureConfig = self.get_feature_config()
#        tfrecord_signature = self.get_tfrecord_signature()
#
#        for num_records in range(1, 250):
#            proto = tf.constant(
#                [feature_config.create_dummy_protobuf(num_records=num_records).SerializeToString()]
#            )
#            try:
#                tfrecord_signature(protos=proto)
#            except Exception:
#                assert False
#
#    def test_serving_required_fields_only(self):
#        """Test serving signature with protos with only required fields"""
#        feature_config: FeatureConfig = self.get_feature_config()
#        tfrecord_signature = self.get_tfrecord_signature()
#
#        proto = tf.constant(
#            [feature_config.create_dummy_protobuf(required_only=True).SerializeToString()]
#        )
#
#        try:
#            tfrecord_signature(protos=proto)
#        except Exception:
#            assert False