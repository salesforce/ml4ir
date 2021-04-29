import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.python.ops import math_ops
import numpy as np
from tensorflow import Tensor
from tensorflow import dtypes

from ml4ir.base.model.metrics.metrics_impl import MetricState
from ml4ir.base.features.feature_config import FeatureConfig

from typing import Optional, Dict


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
        return super(MeanMetricWrapper, self).update_state(
            query_scores, sample_weight=sample_weight
        )


class MeanRankMetric(MeanMetricWrapper):
    def __init__(
        self,
        feature_config: FeatureConfig,
        metadata_features: Dict,
        state: str = MetricState.NEW,
        name="MeanRankMetric",
        dtype: Optional[dtypes.DType] = None,
        **kwargs
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
        **kwargs
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
            **kwargs
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
        **kwargs
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
            **kwargs
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
