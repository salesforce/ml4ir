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

        self.assertTrue(assertion_error_thrown