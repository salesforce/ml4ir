import logging
import unittest

import numpy as np
import tensorflow as tf
import yaml
from ml4ir.applications.ranking.model.metrics.aux_metrics_impl import RankMatchFailure
from ml4ir.base.features.feature_config import FeatureConfig


class RankMachFailureTest(tf.test.TestCase):
    def test_convert_to_rank_scores(self):
        scores = tf.constant([
            [5., 4., 3., 2., 1., -np.inf, -np.inf],
            # Duplicate scores
            [1., 4., 3., 2., 1., -np.inf, -np.inf],
            [5., 4., 3., 2., 1., 6., -np.inf],
            [1., 2., 3., 4., 5., -np.inf, -np.inf],
            [3., 2., 1., 5., 4., -np.inf, -np.inf],
            # 0 scores are retained as 0 rank_scorers
            [3., 2., 1., 5., 4., 0., 0.],
        ])
        actual_rank_scores = RankMatchFailure.convert_to_rank_scores(scores)
        expected_rank_scores = tf.constant([
            [1 / 1., 1 / 2., 1 / 3., 1 / 4., 1 / 5., -np.inf, -np.inf],
            # Note duplicate scores don't get the same rank score
            [1 / 4., 1 / 1., 1 / 2., 1 / 3., 1 / 5., -np.inf, -np.inf],
            [1 / 2., 1 / 3., 1 / 4., 1 / 5., 1 / 6., 1 / 1., -np.inf],
            [1 / 5., 1 / 4., 1 / 3., 1 / 2., 1 / 1., -np.inf, -np.inf],
            [1 / 3., 1 / 4., 1 / 5., 1 / 1., 1 / 2., -np.inf, -np.inf],
            # 0 scores are retained as 0 rank_scorers
            [1 / 3., 1 / 4., 1 / 5., 1 / 1., 1 / 2., 0, 0],
        ])
        tf.debugging.assert_equal(actual_rank_scores, expected_rank_scores)

    def test_normalized_discounted_cumulative_gain(self):
        relevance_grades = tf.constant([
            [1 / 1., 1 / 2., 1 / 3., 1 / 4., 1 / 5., -np.inf, -np.inf],
            [1 / 4., 1 / 1., 1 / 2., 1 / 3., 1 / 5., -np.inf, -np.inf],
            [1 / 2., 1 / 3., 1 / 4., 1 / 5., 1 / 6., 1 / 1., -np.inf],
            [1 / 5., 1 / 4., 1 / 3., 1 / 2., 1 / 1., -np.inf, -np.inf],
            [1 / 3., 1 / 4., 1 / 5., 1 / 1., 1 / 2., 0, 0],
            # This should return -inf as a mask -> not defined in context of rank match failure
            [0., 0., 0., 0., 0., -np.inf, -np.inf],
            [0., 0., 0., 0., 1., -np.inf, -np.inf],
        ])
        ranks = 1 + tf.range(tf.shape(relevance_grades)[1])
        actual_ndcg = RankMatchFailure.normalized_discounted_cumulative_gain(
            relevance_grades, ranks=ranks
        )
        expected_ndcg = tf.constant(
            [1.0, 0.78200406, 0.7245744, 0.62946665, 0.6825818, -np.inf, 0.3868528]
        )
        tf.debugging.assert_equal(actual_ndcg, expected_ndcg)

    def test_discounted_cumulative_gain(self):
        relevance_grades = tf.constant([
            [1 / 1., 1 / 2., 1 / 3., 1 / 4., 1 / 5., -np.inf, -np.inf],
            [1 / 4., 1 / 1., 1 / 2., 1 / 3., 1 / 5., -np.inf, -np.inf],
            [1 / 2., 1 / 3., 1 / 4., 1 / 5., 1 / 6., 1 / 1., -np.inf],
            [1 / 5., 1 / 4., 1 / 3., 1 / 2., 1 / 1., -np.inf, -np.inf],
            [1 / 3., 1 / 4., 1 / 5., 1 / 1., 1 / 2., 0, 0],
            [0., 0., 0., 0., 0., -np.inf, -np.inf],
            [0., 0., 0., 0., 1., -np.inf, -np.inf],
        ])
        ranks = 1 + tf.range(tf.shape(relevance_grades)[1])
        actual_dcg = RankMatchFailure.discounted_cumulative_gain(relevance_grades, ranks=ranks).numpy()
        expected_dcg = np.array(
            [1.5303116, 1.1967099, 1.1404319, 0.9632801, 1.0445628, 0.0, 0.3868528]
        )
        assert np.isclose(actual_dcg, expected_dcg).all()

    def test__compute_match_failure(self):
        y_true_click_rank = tf.constant(
            [
                # Clicked record stays in same position
                4.0,
                4.0,
                4.0,
                4.0,
                # RR improves/degrades
                4.0,
                4.0,
                4.0,
            ]
        )
        y_pred_click_ranks = tf.constant(
            [
                # Clicked record stays in same position
                4.0,
                4.0,
                4.0,
                4.0,
                # RR improves/degrades
                2.0,
                1.0,
                2.0,
            ]
        )
        y_pred_doc_ranks = tf.constant(
            [
                # Clicked record stays in same position
                [1.0, 2.0, 3.0, 4.0, 5.0, np.inf, np.inf],
                [1.0, 2.0, 5.0, 4.0, 3.0, np.inf, np.inf],
                [3.0, 2.0, 5.0, 4.0, 1.0, np.inf, np.inf],
                [3.0, 5.0, 2.0, 4.0, 1.0, np.inf, np.inf],
                # MRR improves/degrades
                [1.0, 4.0, 5.0, 2.0, 3.0, np.inf, np.inf],
                [2.0, 4.0, 5.0, 1.0, 3.0, np.inf, np.inf],
                [1.0, 4.0, 5.0, 2.0, 3.0, np.inf, np.inf],
            ]
        )
        y_aux = tf.constant(
            [
                # Clicked record stays in same position
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 1.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                # MRR improves/degrades
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
            ]
        )
        actual_rmfs = RankMatchFailure._compute_match_failure(
            y_pred_doc_ranks, y_true_click_rank, y_pred_click_ranks, y_aux
        )
        expected_rmfs = tf.constant(
            [0.42372233, 0.3945347, 0.03515863, 0.03515863, 0.36907023, 0.0, 0.36907023]
        )
        self.assertAllClose(actual_rmfs, expected_rmfs, atol=1e-04)

    def test__compute_query_scores(self):
        ranks = tf.constant(
            [
                [1.0, 2.0, 3.0, 4.0, 5.0, -np.inf, -np.inf],
                [1.0, 2.0, 3.0, 4.0, 5.0, -np.inf, -np.inf],
                [1.0, 2.0, 3.0, 4.0, 5.0, -np.inf, -np.inf],
                [1.0, 2.0, 3.0, 4.0, 5.0, -np.inf, -np.inf],
                [1.0, 2.0, 3.0, 4.0, 5.0, -np.inf, -np.inf],
                [1.0, 2.0, 3.0, 4.0, 5.0, -np.inf, -np.inf],
                [1.0, 2.0, 3.0, 4.0, 5.0, -np.inf, -np.inf],
            ]
        )
        y_true = tf.constant(
            [
                # Clicked record stays in same position
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                # RR improves/degrades
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            ]
        )
        y_pred = tf.constant(
            [
                # Clicked record stays in same position
                [0.7, 0.15, 0.07, 0.05, 0.03, 0.0, 0.0],
                [0.7, 0.15, 0.03, 0.05, 0.07, 0.0, 0.0],
                [0.07, 0.15, 0.03, 0.05, 0.7, 0.0, 0.0],
                [0.07, 0.03, 0.15, 0.05, 0.7, 0.0, 0.0],
                # MRR improves/degrades
                [0.7, 0.05, 0.03, 0.15, 0.07, 0.0, 0.0],
                [0.15, 0.05, 0.03, 0.7, 0.07, 0.0, 0.0],
                [0.07, 0.05, 0.03, 0.15, 0.7, 0.0, 0.0],
            ]
        )
        y_aux = tf.constant(
            [
                # Clicked record stays in same position
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 1.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                # MRR improves/degrades
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
                [0.0, 2.0, 3.0, 4.0, 5.0, 0.0, 0.0],
            ]
        )
        mask = tf.constant(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
            ]
        )
        with open(
                "ml4ir/applications/ranking/tests/data/configs/feature_config_aux_loss.yaml"
        ) as feature_config_file:
            feature_config: FeatureConfig = FeatureConfig.get_instance(
                tfrecord_type="sequence",
                feature_config_dict=yaml.safe_load(feature_config_file),
                logger=logging.Logger("test_logger"),
            )
        rmf = RankMatchFailure()
        actual_rmfs = rmf._compute_query_scores(y_true, y_pred, y_aux, ranks, mask)
        expected_rmfs = tf.constant(
            [
                0.42372233,
                0.3945347,
                0.03515863,
                0.03515863,
                0.36907023,
                0.0,
                0.0,
            ]
        )
        self.assertAllClose(actual_rmfs, expected_rmfs, atol=1e-04)


if __name__ == "__main__":
    unittest.main()