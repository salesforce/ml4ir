import os
import numpy as np
import random
import tensorflow as tf

from ml4ir.applications.classification.pipeline import ClassificationPipeline
from ml4ir.applications.classification.tests.test_base import ClassificationTestBase
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.relevance_model import RelevanceModel

class ClassificationModelTest(ClassificationTestBase):
    """
    Test end-to-end model training for classification from model retrieved with
    :func:`~ml4ir.applications.classification.pipeline.ClassificationPipeline.get_relevance_model`
    """

    def run_default_pipeline(self, data_dir: str, data_format: str, feature_config_path: str, model_config_path:str):
        """Train a model with the default set of args"""
        # Fix random seed values for repeatability
        tf.keras.backend.clear_session()
        np.random.seed(123)
        tf.random.set_seed(123)
        random.seed(123)

        args: Namespace = self.args
        # Overriding test default setup args from parameters.
        args.data_dir=data_dir
        args.data_format=data_format
        args.feature_config=feature_config_path
        args.model_config=model_config_path

        classification_pipeline: ClassificationPipeline = ClassificationPipeline(args=args)

        relevance_dataset: RelevanceDataset = classification_pipeline.get_relevance_dataset()
        classification_model: RelevanceModel = classification_pipeline.get_relevance_model()

        classification_model.fit(dataset=relevance_dataset,
                                 num_epochs=5,
                                 models_dir=self.output_dir
        )

        metrics = dict(
            zip(
                classification_model.model.metrics_names,
                classification_model.evaluate(test_dataset=relevance_dataset.test, logs_dir=self.args.logs_dir)
            )
        )

        return metrics

    def test_csv(self):
        """
        Test model training and evaluate the performance metrics from CSV data
        """

        # Test model training on CSV data
        data_dir = os.path.join(self.root_data_dir, "csv")
        feature_config_path = os.path.join(self.root_data_dir, "configs", self.feature_config_fname)
        model_config_path = os.path.join(self.root_data_dir, "configs", self.model_config_fname)

        metrics = self.run_default_pipeline(
            data_dir=data_dir, data_format="csv", feature_config_path=feature_config_path,
            model_config_path=model_config_path
        )

        # Check if the loss and accuracy on the test set is the same
        assert np.isclose(metrics["loss"], 1.966392993927002, rtol=0.01)
        assert np.isclose(metrics["categorical_accuracy"], 0.18229167, rtol=0.01)
        assert np.isclose(metrics["Precision"], 0.2, rtol=0.01)

