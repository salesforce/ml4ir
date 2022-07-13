from typing import Optional, Dict

import numpy as np
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import dtypes
from tensorflow.keras import metrics
from tensorflow.python.ops import math_ops

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.model.metrics.metrics_impl import MetricState, CombinationMetric


class MeanMetricWrapper(metrics.Mean):
    """
    Class that wraps a stateless metric function with the Mean metric.

    Notes
    -----
    Original tensorflow implementation ->
    https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/metrics.py#L541-L590

    MeanMetricWrapper is not a public Class on tf.keras.metrics
    """

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        """
        Create a `MeanMetricWrapper` instance

        Parameters
        ----------
        fn : function
            The metric function to wrap, with signature
            `fn(y_true, y_pred, **kwargs)`.
        name : str, optional
            string name of the metric instance.
        dtype : str, optional
            data type of the metric result.
        **kwargs : dict
            The keyword arguments that are passed on to `fn`.
        """
        super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulates metric statistics by computing the mean of the
        metric function

        Parameters
        ----------
        y_true : Tensor object
            The ground truth values. Shape : [batch_size, max_sequence_size, 1]
        y_pred : Tensor object
            The predicted values. Shape : [batch_size, max_sequence_size, 1]
        sample_weight : Tensor object
            Optional weighting of each example. Defaults to 1. Can be
            a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns
        -------
            Update state of the metric

        Notes
        -----
        `y_true` and `y_pred` should have the same shape.
        """
        query_scores: Tensor = self._fn(y_true, y_pred, **self._fn_kwargs)
        if not sample_weight:
            sample_weight = self.get_sample_weight(query_scores)
        return super(MeanMetricWrapper, self).update_state(
            query_scores, sample_weight=sample_weight
        )

    def get_sample_weight(self, query_scores):
        return None


class MeanRankMetric(MeanMetricWrapper):
    def __init__(
        self,
        feature_config: FeatureConfig,
        metadata_features: Dict,
        state: str = MetricState.NEW,
        name="MeanRankMetric",
        dtype: Optional[dtypes.DType] = None,
        **kwargs,
    ):
        """
        Creates a `MeanRankMetric` instance to compute mean of rank

        Parameters
        ----------
        name : str
            string name of the metric instance.
        dtype : str, optional
            data type of the metric result.
        rank : Tensor object
            2D tensor representing ranks/rankitions of records in a query
        mask : Tensor object
            2D tensor representing 0/1 mask for padded records

        Notes
        -----
        rank and mask should be same shape as y_pred and y_true

        This metric creates two local variables, `total` and `count` that are used to
        compute the frequency with which `y_pred` matches `y_true`. This frequency is
        ultimately returned as `categorical accuracy`: an idempotent operation that
        simply divides `total` by `count`.
        `y_pred` and `y_true` should be passed in as vectors of probabilities, rather
        than as labels. If necessary, use `tf.one_hot` to expand `y_true` as a vector.
        If `sample_weight` is `None`, weights default to 1.
        Use `sample_weight` of 0 to mask values.
        """
        name = "{}_{}".format(state, name)
        # TODO: Handle Example dataset without mask and rank fields
        rank = metadata_features[feature_config.get_rank("node_name")]
        mask = metadata_features[feature_config.get_mask("node_name")]

        super(MeanRankMetric, self).__init__(
            self._compute, name, dtype=dtype, rank=rank, mask=mask
        )
        self.state = state

    def _compute(self, y_true, y_pred, rank, mask):
        """
        Compute mean rank metric

        Parameters
        ----------
        y_true : Tensor object
            Tensor object that contains the true label values
        y_pred : Tensor object
            Tensor object containing the predicted scores
        rank : Tensor object
            Tensor object that contains the rank of each record for a query
        masks : Tensor object
            Tensor object that contains 0/1 flag to identify which
            records were padded and thus should be excluded from metric computation
        """
        if self.state == "new":
            """Rerank using trained model"""
            # Convert y_pred for the masked records to -inf
            y_pred = tf.where(tf.equal(mask, 0), tf.constant(-np.inf), y_pred)

            # Convert predicted ranking scores into ranks for each record per query
            y_pred_ranks = tf.add(
                tf.argsort(
                    tf.argsort(y_pred, axis=-1, direction="DESCENDING", stable=True), stable=True
                ),
                tf.constant(1),
            )

            # Fetch indices of clicked records from y_true
            y_true_clicks = tf.where(tf.equal(tf.cast(y_true, tf.int32), tf.constant(1)))

            # Compute rank of clicked record from predictions
            click_ranks = tf.gather_nd(y_pred_ranks, indices=y_true_clicks)

        else:
            """Compute mean rank metric for existing data"""
            # Fetch indices of clicked records from y_true
            y_true_clicks = tf.where(tf.equal(tf.cast(y_true, tf.int32), tf.constant(1)))

            # Compute rank of clicked record from predictions
            click_ranks = tf.gather_nd(rank, indices=y_true_clicks)

        return self._get_matches_hook(click_ranks)

    def _get_matches_hook(self, y_pred_click_ranks):
        raise NotImplementedError


class MRR(MeanRankMetric):
    """
    Custom metric class to compute the Mean Reciprocal Rank.

    Calculates the mean of the reciprocal ranks of the
    clicked records from a list of queries.

    Examples
    --------
    >>> `y_true` is [[0, 0, 1], [0, 1, 0]]
    >>> `y_pred` is [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> `mask` is [[1, 1, 1], [1, 1, 1]] and
    >>> `rank` is [[1, 3, 2], [3, 1, 2]]
    >>> then the MRR is 0.75
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        metadata_features: Dict,
        name="MRR",
        state=MetricState.NEW,
        **kwargs,
    ):
        """
        Creates a `MRR` instance to compute mean of reciprocal rank

        Parameters
        ----------
        feature_config : FeatureConfig object
            FeatureConfig object that defines the configuration for each model
            feature
        metadata_features : dict
            Dictionary of metadata feature tensors that can be used to compute
            custom metrics
        name : str
            Name of the metric
        state : {"new", "old"}
            State of the metric
        """
        super(MRR, self).__init__(
            feature_config=feature_config,
            metadata_features=metadata_features,
            name=name,
            state=state,
            **kwargs,
        )

    def _get_matches_hook(self, y_pred_click_ranks):
        """
        Return reciprocal click ranks for MRR

        Parameters
        ----------
        y_pred_click_ranks: Tensor object
            Tensor object containing the ranks of the clicked records for each query

        Returns
        -------
        Tensor object
            Reciprocal ranks tensor
        """
        return math_ops.reciprocal(tf.cast(y_pred_click_ranks, tf.float32))


class ACR(MeanRankMetric):
    """
    Custom metric class to compute the Average Click Rank.

    Calculates the mean of the ranks of the
    clicked records from a list of queries.

    Examples
    --------
    >>> `y_true` is [[0, 0, 1], [0, 1, 0]]
    >>> `y_pred` is [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> `mask` is [[1, 1, 1], [1, 1, 1]] and
    >>> `rank` is [[1, 3, 2], [3, 1, 2]]
    >>> then the ACR is 1.50
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        metadata_features: Dict,
        name="ACR",
        state=MetricState.NEW,
        **kwargs,
    ):
        """
        Creates a `ACR` instance to compute mean of rank

        Parameters
        ----------
        feature_config : FeatureConfig object
            FeatureConfig object that defines the configuration for each model
            feature
        metadata_features : dict
            Dictionary of metadata feature tensors that can be used to compute
            custom metrics
        name : str
            Name of the metric
        state : {"new", "old"}
            State of the metric
        """
        super(ACR, self).__init__(
            feature_config=feature_config,
            metadata_features=metadata_features,
            name=name,
            state=state,
            **kwargs,
        )

    def _get_matches_hook(self, y_pred_click_ranks):
        """
        Return click ranks for ACR

        Parameters
        ----------
        y_pred_click_ranks: Tensor object
            Tensor object containing the ranks of the clicked records for each query

        Returns
        -------
        Tensor object
            Ranks tensor cast to float
        """
        return tf.cast(y_pred_click_ranks, tf.float32)


class RankMatchFailure(MeanMetricWrapper, CombinationMetric):
    def __init__(
        self,
        feature_config: FeatureConfig,
        metadata_features: Dict,
        state: str = MetricState.NEW,
        name="RankMatchFailure",
        dtype: Optional[dtypes.DType] = None,
        **kwargs,
    ):
        """
        Creates a `RankMatchFailure` instance to compute mean of rank

        Parameters
        ----------
        name : str
            string name of the metric instance.
        dtype : str, optional
            data type of the metric result.
        rank : Tensor object
            2D tensor representing ranks/rankitions of records in a query
        mask : Tensor object
            2D tensor representing 0/1 mask for padded records

        Notes
        -----
        rank and mask should be same shape as y_pred and y_true

        This metric creates two local variables, `total` and `count` that are used to
        compute the frequency with which `y_pred` matches `y_true`. This frequency is
        ultimately returned as `categorical accuracy`: an idempotent operation that
        simply divides `total` by `count`.
        `y_pred` and `y_true` should be passed in as vectors of probabilities, rather
        than as labels. If necessary, use `tf.one_hot` to expand `y_true` as a vector.
        If `sample_weight` is `None`, weights default to 1.
        Use `sample_weight` of 0 to mask values.
        """
        name = "{}_{}".format(state, name)
        # TODO: Handle Example dataset without mask and rank fields
        rank = metadata_features[feature_config.get_rank("node_name")]
        mask = metadata_features[feature_config.get_mask("node_name")]
        if not feature_config.aux_label:
            raise ValueError(
                f"{self.__class__.__qualname__} needs an aux label in the feature config"
            )
        y_aux = metadata_features[feature_config.get_aux_label("node_name")]

        super().__init__(self._compute, name, dtype=dtype, y_aux=y_aux, rank=rank, mask=mask)
        self.state = state

    def get_sample_weight(self, query_scores):
        mask = tf.ones_like(query_scores)
        return tf.where(
            query_scores == tf.constant(-np.inf, dtype=tf.float32),
            tf.constant(0, dtype=tf.float32),
            mask,
        )

    def _compute(self, y_true, y_pred, y_aux, rank, mask):
        """
        Compute mean rank metric

        Parameters
        ----------
        y_true : Tensor object
            Tensor object that contains the true label values
        y_pred : Tensor object
            Tensor object containing the predicted scores
        rank : Tensor object
            Tensor object that contains the rank of each record for a query
        mask : Tensor object
            Tensor object that contains 0/1 flag to identify which
            records were padded and thus should be excluded from metric computation
        """
        if self.state == "new":
            """Rerank using trained model"""
            # Convert y_pred for the masked records to -inf
            y_pred = tf.where(tf.equal(mask, 0), tf.constant(-np.inf), y_pred)

            # Convert predicted ranking scores into ranks for each record per query
            # TODO: Currently these ranks are defined below the clicked document too. Scores below the clicked document shouldn't affect the final rank for NDCG
            y_pred_ranks = tf.add(
                tf.argsort(
                    tf.argsort(y_pred, axis=-1, direction="DESCENDING", stable=True), stable=True
                ),
                tf.constant(1),
            )
            click_ranks = tf.reduce_sum(
                tf.where(tf.equal(tf.cast(y_true, tf.int32), tf.constant(1)), y_pred_ranks, 0),
                axis=-1,
            )

            y_true_click_rank = tf.reduce_sum(
                tf.where(tf.equal(tf.cast(y_true, tf.int32), tf.constant(1)), rank, 0), axis=-1
            )
            ranks = y_pred_ranks

        else:
            """Compute mean rank metric for existing data"""
            y_true_click_rank = tf.reduce_sum(
                tf.where(tf.equal(tf.cast(y_true, tf.int32), tf.constant(1)), rank, 0), axis=-1
            )
            click_ranks = y_true_click_rank
            ranks = rank
        # Mask ranks with max possible value so that they are ignored downstream
        ranks = tf.where(tf.equal(mask, 0), tf.constant(np.inf), tf.cast(ranks, tf.float32))

        return self._compute_match_failure(
            tf.cast(ranks, tf.float32),
            tf.cast(y_true_click_rank, tf.float32),
            tf.cast(click_ranks, tf.float32),
            tf.cast(y_aux, tf.float32),
        )

    @staticmethod
    def _compute_match_failure(ranks, y_true_click_rank, metric_click_ranks, y_aux):
        """
        Compute match failure metric for a batch

        Parameters
        ----------
        ranks : Tensor object
            Tensor object that contains scores for various documents
        y_true_click_rank : Tensor object
            Tensor object that contains scores for various documents
        metric_click_ranks : Tensor object
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
            ranks <= tf.expand_dims(metric_click_ranks, axis=-1), y_aux, tf.constant(-np.inf)
        )
        rank_scores = RankMatchFailure.convert_to_rank_scores(scores)
        ranks_above_click = tf.where(
            ranks <= tf.expand_dims(metric_click_ranks, axis=-1), ranks, tf.constant(np.inf)
        )
        num_match = tf.cast(tf.math.count_nonzero(scores > 0, axis=-1), tf.float32)
        match_failure = 1 - tf.cast(
            RankMatchFailure.normalized_discounted_cumulative_gain(rank_scores, ranks_above_click),
            tf.float32,
        )
        # If all records<=click have a name match, then it is not an NMF
        # If number of scores>0 is same as the clicked rank, all ranks have a name match
        match_failure = tf.where(
            tf.equal(metric_click_ranks, num_match),
            tf.constant(0, dtype=tf.float32),
            match_failure,
        )
        # No Match Failure when there is no match on the clicked rank
        idxs = tf.expand_dims(tf.range(tf.shape(ranks)[0]), -1)
        y_true_click_rank = tf.expand_dims(tf.cast(y_true_click_rank, tf.int32), axis=-1)
        y_true_click_idx = tf.where(y_true_click_rank > 0, y_true_click_rank - 1, 0)
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
