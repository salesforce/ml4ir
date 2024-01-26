import unittest

import numpy as np

from ml4ir.base.model.layers.robust_scaler import RobustScalerLayer


class TestRobustScalerLayer(unittest.TestCase):

    def test_robust_scaler_default(self):
        input_feature = np.array([[0.0, 0.8, 0.9, 0.5, 0.6, 0.7]])

        percentile_25 = np.percentile(input_feature, 25)
        percentile_75 = np.percentile(input_feature, 75)

        robust_scaler_op = RobustScalerLayer(p25=percentile_25, p75=percentile_75)

        actual_reciprocal_ranks = robust_scaler_op(input_feature).numpy()
        expected_reciprocal_ranks = (input_feature - percentile_25) / (percentile_75 - percentile_25)

        self.assertTrue(np.isclose(actual_reciprocal_ranks, expected_reciprocal_ranks).all())

    def test_robust_scaler_negative_scores(self):
        input_feature = np.array([[0.0, 0.8, 0.9, -0.5, -0.4, -0.3]])

        percentile_25 = np.percentile(input_feature, 25)
        percentile_75 = np.percentile(input_feature, 75)

        robust_scaler_op = RobustScalerLayer(p25=percentile_25, p75=percentile_75)

        actual_reciprocal_ranks = robust_scaler_op(input_feature).numpy()
        expected_reciprocal_ranks = (input_feature - percentile_25) / (percentile_75 - percentile_25)

        self.assertTrue(np.isclose(actual_reciprocal_ranks, expected_reciprocal_ranks).all())

    def test_robust_scaler_zeros(self):
        input_feature = np.array([[0.0, 0.0, 0.0, 0.0]])

        percentile_25 = np.percentile(input_feature, 25)
        percentile_75 = np.percentile(input_feature, 75)

        robust_scaler_op = RobustScalerLayer(p25=percentile_25, p75=percentile_75)

        actual_reciprocal_ranks = robust_scaler_op(input_feature).numpy()
        expected_reciprocal_ranks = input_feature

        self.assertTrue(np.isclose(actual_reciprocal_ranks, expected_reciprocal_ranks).all())