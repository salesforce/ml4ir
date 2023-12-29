import unittest

import numpy as np
from scipy.stats import zscore

from ml4ir.applications.ranking.model.layers.normalization import QueryNormalization, TheoreticalMinMaxNormalization, \
    ReciprocalRankLayer


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


class TestReciprocalRank(unittest.TestCase):

    def test_reciprocal_rank_default(self):
        reciprocal_rank_op = ReciprocalRankLayer()
        input_feature = np.array([[0.0, 0.8, 0.9, 0.5, 0.6, 0.7]])

        actual_reciprocal_ranks = reciprocal_rank_op(input_feature).numpy()

        expected_reciprocal_ranks = np.array([[0., 1. / 2, 1. / 1, 1. / 5, 1. / 4, 1. / 3]])
        self.assertTrue(np.isclose(actual_reciprocal_ranks, expected_reciprocal_ranks).all())

    def test_reciprocal_rank_2d_vs_3d(self):
        reciprocal_rank_op = ReciprocalRankLayer()
        input_feature = np.array([[0.0, 0.8, 0.9, 0.5, 0.6, 0.7]])

        reciprocal_ranks_2d = reciprocal_rank_op(input_feature).numpy()
        reciprocal_ranks_3d = reciprocal_rank_op(input_feature[:, :, np.newaxis]).numpy()

        self.assertTrue(np.isclose(np.squeeze(reciprocal_ranks_2d), np.squeeze(reciprocal_ranks_3d)).all())

    def test_reciprocal_rank_k(self):
        reciprocal_rank_op = ReciprocalRankLayer(k=60)
        input_feature = np.array([[0.0, 0.8, 0.9, 0.5, 0.6, 0.7]])

        actual_reciprocal_ranks = reciprocal_rank_op(input_feature).numpy()

        expected_reciprocal_ranks = np.array([[0., 1. / (60. + 2), 1. / (60. + 1), 1. / (60. + 5), 1. / (60. + 4), 1. / (60. + 3)]])
        self.assertTrue(np.isclose(actual_reciprocal_ranks, expected_reciprocal_ranks).all())

    def test_reciprocal_rank_k_trainable(self):
        reciprocal_rank_op = ReciprocalRankLayer()
        self.assertEquals(len(reciprocal_rank_op.trainable_variables), 0)
        self.assertEquals(len(reciprocal_rank_op.non_trainable_variables), 1)

        reciprocal_rank_op = ReciprocalRankLayer(k_trainable=True)
        self.assertEquals(len(reciprocal_rank_op.trainable_variables), 1)
        self.assertEquals(len(reciprocal_rank_op.non_trainable_variables), 0)

    def test_reciprocal_rank_ignore_zero_score(self):
        reciprocal_rank_op = ReciprocalRankLayer(ignore_zero_score=False)
        input_feature = np.array([[0.0, 0.8, 0.9, 0.5, 0.6, 0.7]])

        actual_reciprocal_ranks = reciprocal_rank_op(input_feature).numpy()

        expected_reciprocal_ranks = np.array([[1. / 6, 1. / 2, 1. / 1, 1. / 5, 1. / 4, 1. / 3]])
        self.assertTrue(np.isclose(actual_reciprocal_ranks, expected_reciprocal_ranks).all())

    def test_reciprocal_rank_negative_scores(self):
        reciprocal_rank_op = ReciprocalRankLayer(ignore_zero_score=False)
        input_feature = np.array([[0.0, 0.8, 0.9, -0.5, -0.4, -0.3]])

        actual_reciprocal_ranks = reciprocal_rank_op(input_feature).numpy()

        expected_reciprocal_ranks = np.array([[1. / 3, 1. / 2, 1. / 1, 1. / 6, 1. / 5, 1. / 4]])
        self.assertTrue(np.isclose(actual_reciprocal_ranks, expected_reciprocal_ranks).all())
