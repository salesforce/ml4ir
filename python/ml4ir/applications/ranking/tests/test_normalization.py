import unittest

import numpy as np
from scipy.stats import zscore

from ml4ir.applications.ranking.model.layers.normalization import QueryNormalization, \
    TheoreticalMinMaxNormalization, EstimatedMinMaxNormalization


class TestQueryNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.query_norm = QueryNormalization(requires_mask=True)

    def test_query_norm_requires_mask(self):
        """Test that the layer fails if requires_mask is not set to True in the args"""
        assertion_error_thrown = False
        try:
            QueryNormalization()
        except AssertionError:
            assertion_error_thrown = True

        self.assertTrue(assertion_error_thrown)

    def test_query_norm_with_no_mask(self):
        """Test query normalization with zscore when no records are masked"""
        input = np.random.randn(1, 5, 8)
        mask = np.array([[1, 1, 1, 1, 1]])

        actual_output = self.query_norm(input, mask)
        expected_output = zscore(input, axis=1)

        self.assertTrue(np.isclose(actual_output, expected_output, atol=1e-05).all())

    def test_query_norm_with_mask(self):
        """Test query normalization with zscore when some records are masked"""
        input = np.random.randn(1, 5, 8)
        mask = np.array([[1, 1, 1, 0, 0]])

        actual_output = self.query_norm(input, mask).numpy()
        expected_output = zscore(input[np.where(mask)], axis=0)

        # Test the values of the unmasked records
        self.assertTrue(np.isclose(actual_output[np.where(mask)], expected_output, atol=1e-05).all())

        # Test the values of the masked records
        self.assertTrue(actual_output[np.where(mask == 0.)].sum() == 0.)


class TestEstimatedMinMaxNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.emm_norm = EstimatedMinMaxNormalization(requires_mask=True)

    def test_emm_norm_requires_mask(self):
        """Test that the layer fails if requires_mask is not set to True in the args"""
        assertion_error_thrown = False
        try:
            EstimatedMinMaxNormalization()
        except AssertionError:
            assertion_error_thrown = True

        self.assertTrue(assertion_error_thrown)

    def test_emm_norm_with_no_mask(self):
        """Test EMM normalization when no records are masked"""
        input = np.random.randn(1, 5, 8)
        mask = np.array([[1, 1, 1, 1, 1]])

        actual_output = self.emm_norm(input, mask)
        expected_min = (np.mean(input, axis=1) - 3 * np.std(input, axis=1))
        expected_max = (np.mean(input, axis=1) + 3 * np.std(input, axis=1))
        expected_output = (input - expected_min) / (expected_max - expected_min)

        self.assertTrue(np.isclose(actual_output, expected_output, atol=1e-05).all())

    def test_emm_norm_with_mask(self):
        """Test EMM normalization with zscore when some records are masked"""
        input = np.random.randn(1, 5, 8)
        mask = np.array([[1, 1, 1, 0, 0]])

        actual_output = self.emm_norm(input, mask).numpy()
        expected_min = (np.mean(input[:, :3, :], axis=1) - 3 * np.std(input[:, :3, :], axis=1))
        expected_max = (np.mean(input[:, :3, :], axis=1) + 3 * np.std(input[:, :3, :], axis=1))
        expected_output = (input[:, :3, :] - expected_min) / (expected_max - expected_min)

        # Test the values of the unmasked records
        self.assertTrue(np.isclose(actual_output[:, :3, :], expected_output, atol=1e-05).all())

        # Test the values of the masked records
        self.assertTrue(actual_output[np.where(mask == 0.)].sum() == 0.)

    def test_emm_norm_with_multiple_queries(self):
        """Test EMM normalization when there are multiple queries"""
        input = np.random.randn(2, 5, 8)
        mask = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])

        actual_output = self.emm_norm(input, mask)

        expected_min_query_0 = (np.mean(input[0, :, :], axis=0) - 3 * np.std(input[0, :, :], axis=0))
        expected_max_query_0 = (np.mean(input[0, :, :], axis=0) + 3 * np.std(input[0, :, :], axis=0))
        expected_output_query_0 = (input[0, :, :] - expected_min_query_0) / (expected_max_query_0 - expected_min_query_0)

        self.assertTrue(np.isclose(actual_output[0, :, :], expected_output_query_0, atol=1e-05).all())

        expected_min_query_1 = (np.mean(input[1, :, :], axis=0) - 3 * np.std(input[1, :, :], axis=0))
        expected_max_query_1 = (np.mean(input[1, :, :], axis=0) + 3 * np.std(input[1, :, :], axis=0))
        expected_output_query_1 = (input[1, :, :] - expected_min_query_1) / (expected_max_query_1 - expected_min_query_1)

        self.assertTrue(np.isclose(actual_output[1 :, :], expected_output_query_1, atol=1e-05).all())

    def test_emm_norm_with_one_result(self):
        """Test EMM normalization when a query has only 1 result"""
        input = np.random.randn(1, 1, 8)
        mask = np.array([[1]])

        actual_output = self.emm_norm(input, mask)

        self.assertTrue(np.isclose(actual_output, np.zeros_like(input), atol=1e-05).all())

class TestTheoreticalMinMaxNormalization(unittest.TestCase):
    def setUp(self) -> None:
        self.tmm_norm = TheoreticalMinMaxNormalization(theoretical_min=0.5)

    def test_tmm_norm_with_2d_feature(self):
        """Test query normalization with zscore when some records are masked"""
        input_feature = np.array([[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

        actual_normed_feature = self.tmm_norm(input_feature).numpy()

        expected_normed_feature = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1.]])

        self.assertTrue(np.isclose(actual_normed_feature, expected_normed_feature).all())
        self.assertTrue(input_feature.shape == actual_normed_feature.shape)

    def test_tmm_norm_with_3d_feature(self):
        """Test query normalization with zscore when some records are masked"""
        # Create a 3d feature with last dimension 3
        input_feature = np.repeat(
            np.array([-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])[np.newaxis, :, np.newaxis], 3,
            axis=2)

        actual_normed_feature = self.tmm_norm(input_feature).numpy()

        expected_normed_feature = np.repeat(
            np.array([0., 0., 0., 0., 0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1.])[np.newaxis, :, np.newaxis], 3, axis=2)

        self.assertTrue(np.isclose(actual_normed_feature, expected_normed_feature).all())
        self.assertTrue(input_feature.shape == actual_normed_feature.shape)
