import unittest

import numpy as np
from scipy.stats import zscore
import tensorflow as tf

from ml4ir.applications.ranking.model.layers.normalization import QueryNormalization


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

        self.assertTrue(np.isclose(actual_output, expected_output).all())

    def test_query_norm_with_mask(self):
        """Test query normalization with zscore when some records are masked"""
        input = np.random.randn(1, 5, 8)
        mask = np.array([[1, 1, 1, 0, 0]])

        actual_output = self.query_norm(input, mask).numpy()
        expected_output = zscore(input[np.where(mask)], axis=0)

        # Test the values of the unmasked records
        self.assertTrue(np.isclose(actual_output[np.where(mask)], expected_output).all())

        # Test the values of the masked records
        self.assertTrue(actual_output[np.where(mask == 0.)].sum() == 0.)
