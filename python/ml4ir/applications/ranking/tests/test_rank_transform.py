import unittest

import numpy as np

from ml4ir.applications.ranking.model.layers.rank_transform import ReciprocalRankLayer


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

    def test_reciprocal_rank_scale_range_to_one(self):
        input_feature = np.array([[0.0, 0.8, 0.9, 0.5, 0.6, 0.7]])

        reciprocal_rank_op = ReciprocalRankLayer(scale_range_to_one=True)
        actual_reciprocal_ranks = reciprocal_rank_op(input_feature).numpy()
        expected_reciprocal_ranks = np.array([[0., 1. / 2, 1. / 1, 1. / 5, 1. / 4, 1. / 3]])
        self.assertTrue(np.isclose(actual_reciprocal_ranks, expected_reciprocal_ranks).all())

        reciprocal_rank_op = ReciprocalRankLayer(k=60, scale_range_to_one=True)
        actual_reciprocal_ranks = reciprocal_rank_op(input_feature).numpy()
        expected_reciprocal_ranks = np.array([[0., (60 + 1.) / (60. + 2), (60 + 1.) / (60. + 1), (60 + 1.) / (60. + 5), (60 + 1.) / (60. + 4), (60 + 1.) / (60. + 3)]])
        self.assertTrue(np.isclose(actual_reciprocal_ranks, expected_reciprocal_ranks).all())
