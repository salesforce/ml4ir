import numpy as np
from argparse import Namespace

from ml4ir.applications.classification.pipeline import ClassificationPipeline
from ml4ir.applications.classification.tests.test_base import ClassificationTestBase
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.relevance_model import RelevanceModel


class ClassificationModelTest(ClassificationTestBase):
    """
    Test end-to-end model training for classification from model retrieved with
    :func:`~ml4ir.applications.classification.pipeline.ClassificationPipeline.get_relevance_model`
    """

    def run_default_pipeline(self, data_format: str):
        """Train a model with the default set of args"""
        # Fix random seed values for repeatability
        self.set_seeds()

        args: Namespace = self.get_overridden_args(data_format)

        classification_pipeline: ClassificationPipeline = ClassificationPipeline(args=args)
        relevance_dataset: RelevanceDataset = classification_pipeline.get_relevance_dataset()
        classification_model: RelevanceModel = classification_pipeline.get_relevance_model()

        classification_model.fit(
            dataset=relevance_dataset, num_epochs=5, models_dir=self.output_dir
        )

        _, _, metrics = classification_model.evaluate(
            test_dataset=relevance_dataset.test, logs_dir=self.args.logs_dir
        )

        return metrics

    def test_csv(self):
        """
        Test model training and evaluate the performance metrics from CSV data
        """
        # Test model training on CSV data
        metrics = self.run_default_pipeline(data_format="csv")

        # Check if the loss and accuracy on the test set is the same
        assert np.isclose(metrics["loss"], 1.966392993927002, rtol=0.01)
        assert np.isclose(metrics["categorical_accuracy"], 0.18229167, rtol=0.01)
        assert np.isclose(metrics["Precision"], 0.2, rtol=0.01)
