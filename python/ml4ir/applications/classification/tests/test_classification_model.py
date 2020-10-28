import numpy as np
from ml4ir.applications.classification.tests.test_base import ClassificationTestBase


class ClassificationModelTest(ClassificationTestBase):
    """
    Test end-to-end model training for classification from model retrieved with
    :func:`~ml4ir.applications.classification.pipeline.ClassificationPipeline.get_relevance_model`
    """

    def test_csv_metrics(self):
        """
        Test the performance metrics from CSV data
        """
        # Check if the loss and accuracy on the test set is the same
        # Note that we don't check Precision which is not useful for this test model
        # Note that these numbers are different if you run it directly vs with docker-compose up
        expected_loss = 2.205
        expected_acc = 0.083
        tol = 0.01
        self.assertTrue(np.isclose(self.metrics_dict["loss"], expected_loss, rtol=tol),
                        msg=f"Loss not in expected range."
                            f" Expected: {expected_loss} ± {tol}, Found: {self.metrics_dict['loss']}")
        self.assertTrue(np.isclose(self.metrics_dict["categorical_accuracy"], expected_acc, rtol=tol),
                        msg=f"Categorical_accuracy not in expected range."
                            f" Expected: {expected_acc} ± {tol}, Found: {self.metrics_dict['categorical_accuracy']}")

    def test_group_metrics_df(self):
        """
        Test the dimensions of the grouped metrics
        """
        metrics = self.classification_pipeline.metrics_keys  # Metrics during training
        self.assertTrue(self.grouped_metrics.metric.nunique() == len(metrics))  # number of metrics
        self.assertTrue(set(self.grouped_metrics.metric.unique()) == set(metrics))  # metrics per se
        group_keys = self.classification_pipeline.feature_config.get_group_metrics_keys()
        group_names = [key['name'] for key in group_keys]
        self.assertTrue(set(self.grouped_metrics.group_name.unique()) == set(group_names))  # group_names
