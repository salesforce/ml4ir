import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics


class RankMatchFailure(metrics.Mean):
    """Custom metric implementation to compute the ranking performance on an auxiliary label"""

    def update_state(self, y_true, y_pred, y_aux, y_true_ranks, mask, sample_weight=None):
        """
        Accumulates metric statistics by computing the mean of the
        metric function

        Parameters
        ----------
        y_true : Tensor object
            Tensor object that contains the true label values
        y_pred : Tensor object
            Tensor object containing the predicted scores
        y_aux : Tensor object
            Tensor object that contains the values for the auxiliary label
        y_true_ranks : Tensor object
            Tensor object that contains the rank of each record for a query
        mask : Tensor object
            Tensor object that contains 0/1 flag to identify which
            records were padded and thus should be excluded from metric computation

        Returns
        -------
            Updated state of the metric

        Notes
        -----
        `y_aux` is a mandatory argument as this is a metric designed for the auxiliary label
        """
        query_scores = self._compute_query_scores(y_true, y_pred, y_aux, y_true_ranks, mask)
        sample_weight = self._get_sample_weight(query_scores)

        return super().update_state(query_scores, sample_weight)

    def _compute_query_scores(self, y_true, y_pred, y_aux, y_true_ranks, mask):
        """
        Compute the auxiliary RankMatchFailure score for each query that will be aggregated via a mean upstream

        Parameters
        ----------
        y_true : Tensor object
            Tensor object that contains the true label values
        y_pred : Tensor object
            Tensor object containing the predicted scores
        y_aux : Tensor object
            Tensor object that contains the values for the auxiliary label
        y_true_ranks : Tensor object
            Tensor object that contains the rank of each record for a query
        mask : Tensor object
            Tensor object that contains 0/1 flag to identify which
            records were padded and thus should be excluded from metric computation

        Returns
        -------
        Tensor
            Score for each query
        """
        # Convert y_pred for the masked records to -inf
        y_pred = tf.where(tf.equal(mask, 0), tf.constant(-np.inf), y_pred)

        # Convert predicted ranking scores into ranks for each record per query
        # TODO: Currently these ranks are defined below the clicked document too.
        #       Scores below the clicked document shouldn't affect the final rank for NDCG
        y_pred_ranks = tf.add(
            tf.argsort(
                tf.argsort(y_pred, axis=-1, direction="DESCENDING", stable=True), stable=True
            ),
            tf.constant(1),
        )
        y_pred_click_ranks = tf.reduce_sum(
            tf.where(tf.equal(tf.cast(y_true, tf.int32), tf.constant(1)), y_pred_ranks, 0),
            axis=-1,
        )

        # Compute original rank of the clicked record
        y_true_click_ranks = tf.reduce_sum(
            tf.where(tf.equal(tf.cast(y_true, tf.int32), tf.constant(1)), y_true_ranks, 0), axis=-1
        )

        # Mask ranks with max possible value so that they are ignored downstream
        ranks = tf.where(tf.equal(mask, 0), tf.constant(np.inf), tf.cast(y_pred_ranks, tf.float32))

        return self._compute_match_failure(
            tf.cast(ranks, tf.float32),
            tf.cast(y_true_click_ranks, tf.float32),
            tf.cast(y_pred_click_ranks, tf.float32),
            tf.cast(y_aux, tf.float32),
        )

    @staticmethod
    def _get_sample_weight(query_scores):
        mask = tf.ones_like(query_scores)
        return tf.where(
            query_scores == tf.constant(-np.inf, dtype=tf.float32),
            tf.constant(0, dtype=tf.float32),
            mask,
        )

    @staticmethod
    def _compute_match_failure(ranks, y_true_click_ranks, y_pred_click_ranks, y_aux):
        """
        Compute match failure metric for a batch
        Parameters
        ----------
        ranks : Tensor object
            Tensor object that contains scores for various documents
        y_true_click_ranks : Tensor object
            Tensor object that contains scores for various documents
        y_pred_click_ranks : Tensor object
            Tensor object that contains scores for various documents
        y_aux : Tensor object
            Tensor object that contains scores for various documents
        Returns
        -------
        Tensor object
            Tensor of Match Failure scores for each query
        """
        # Mask all values of y_aux which are below the clicked rank
        scores = tf.where(
            ranks <= tf.expand_dims(y_pred_click_ranks, axis=-1), y_aux, tf.constant(-np.inf)
        )
        rank_scores = RankMatchFailure.convert_to_rank_scores(scores)
        ranks_above_click = tf.where(
            ranks <= tf.expand_dims(y_pred_click_ranks, axis=-1), ranks, tf.constant(np.inf)
        )
        num_match = tf.cast(tf.math.count_nonzero(scores > 0, axis=-1), tf.float32)
        match_failure = 1 - tf.cast(
            RankMatchFailure.normalized_discounted_cumulative_gain(rank_scores, ranks_above_click),
            tf.float32,
        )
        # If all records<=click have a name match, then it is not an NMF
        # If number of scores>0 is same as the clicked rank, all ranks have a name match
        match_failure = tf.where(
            tf.equal(y_pred_click_ranks, num_match),
            tf.constant(0, dtype=tf.float32),
            match_failure,
        )
        # No Match Failure when there is no match on the clicked rank
        idxs = tf.expand_dims(tf.range(tf.shape(ranks)[0]), -1)
        y_true_click_ranks = tf.expand_dims(tf.cast(y_true_click_ranks, tf.int32), axis=-1)
        y_true_click_idx = tf.where(y_true_click_ranks > 0, y_true_click_ranks - 1, 0)
        clicked_records_score = tf.gather_nd(
            y_aux, indices=tf.concat([idxs, y_true_click_idx], axis=-1)
        )
        match_failure = tf.where(
            clicked_records_score == 0.0, tf.constant(-np.inf, dtype=tf.float32), match_failure
        )
        return match_failure

    @staticmethod
    def convert_to_rank_scores(scores):
        """
        Maps each score -> 1/rank for standardizing the score ranges across queries
        Parameters
        ----------
        scores : Tensor object
            Tensor object that contains scores for various documents
        Returns
        -------
        Tensor object
            Tensor of 1/rank(score)
        """
        # Note: 2 scores with same value can get different rank scores.
        # There is no sane TF way to handle this today
        scores = tf.cast(scores, dtype=tf.float32)
        score_rank = tf.add(
            tf.argsort(
                tf.argsort(scores, axis=-1, direction="DESCENDING", stable=True), stable=True
            ),
            tf.constant(1),
        )
        rank_scores = 1 / score_rank
        rank_scores = tf.cast(rank_scores, dtype=tf.float32)
        rank_scores = tf.where(
            scores == tf.constant(0.0, dtype=tf.float32),
            tf.constant(0.0, dtype=tf.float32),
            rank_scores,
            )
        # -inf is used as mask
        rank_scores = tf.where(
            scores == tf.constant(-np.inf, dtype=tf.float32),
            tf.constant(-np.inf, dtype=tf.float32),
            rank_scores,
            )
        return rank_scores

    @staticmethod
    def discounted_cumulative_gain(relevance_grades, ranks):
        """
        Compute the discounted cumulative gain
        Parameters
        ----------
        relevance_grades : Tensor object
            Tensor object that contains scores in the order of ranks (predicted or ideal)
        Returns
        -------
        Tensor object
            Tensor of DCG scores along 0th axis
        """
        dcg_unmasked = (tf.cast(tf.math.pow(2.0, relevance_grades) - 1, dtype=tf.float32)) / (
                tf.math.log(tf.cast(ranks, dtype=tf.float32) + 1) / tf.math.log(2.0)
        )
        # Remove DCG where relevance grade was -inf (mask)
        dcg_masked = tf.where(dcg_unmasked < 0, tf.constant(0.0, dtype=tf.float32), dcg_unmasked)
        return tf.reduce_sum(dcg_masked, axis=-1)

    @staticmethod
    def normalized_discounted_cumulative_gain(relevance_grades, ranks):
        """
        Compute the normalized discounted cumulative gain
        Parameters
        ----------
        relevance_grades : Tensor object
            Tensor object that contains scores in the order of predicted ranks
        Returns
        -------
        Tensor object
            Tensor of NDCG scores along 0th axis
        """
        ideal_ranks = 1 + tf.range(tf.shape(relevance_grades)[1])
        sorted_relevance_grades = tf.sort(relevance_grades, direction="DESCENDING", axis=-1)
        dcg_score = RankMatchFailure.discounted_cumulative_gain(relevance_grades, ranks)
        idcg_score = RankMatchFailure.discounted_cumulative_gain(
            sorted_relevance_grades, ideal_ranks
        )
        ndcg_raw = dcg_score / idcg_score
        return tf.where(
            idcg_score == 0, tf.constant(-np.inf, dtype=tf.float32), ndcg_raw
        )  # Handle invalid condition, return -inf
