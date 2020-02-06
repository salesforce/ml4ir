import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.python.ops import math_ops
import numpy as np
from tensorflow import Tensor
from tensorflow.dtypes import DType
from typing import Optional


class MeanMetricWrapper(metrics.Mean):
    """
    Wraps a stateless metric function with the Mean metric.

    Original tensorflow implementation ->
    https://github.com/tensorflow/tensorflow/blob/r2.0/tensorflow/python/keras/metrics.py#L541-L590

    NOTE: MeanMetricWrapper is not a public Class on tf.keras.metrics
    """

    def __init__(self, fn, name=None, dtype=None, **kwargs):
        """Creates a `MeanMetricWrapper` instance.

        Args:
          fn: The metric function to wrap, with signature
            `fn(y_true, y_pred, **kwargs)`.
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
          **kwargs: The keyword arguments that are passed on to `fn`.
        """
        super(MeanMetricWrapper, self).__init__(name=name, dtype=dtype)
        self._fn = fn
        self._fn_kwargs = kwargs

    def update_state(self, y_true, y_pred, sample_weight=None):
        """Accumulates metric statistics.

        `y_true` and `y_pred` should have the same shape.

        Args:
          y_true: The ground truth values.
          y_pred: The predicted values.
          sample_weight: Optional weighting of each example. Defaults to 1. Can be
            a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns:
          Update op.
        """
        query_scores: Tensor = self._fn(y_true, y_pred, **self._fn_kwargs)
        return super(MeanMetricWrapper, self).update_state(
            query_scores, sample_weight=sample_weight
        )


class MeanRankMetric(MeanMetricWrapper):
    def __init__(
        self, pos: Tensor, mask: Tensor, name="MeanRankMetric", dtype: Optional[DType] = None,
    ):
        """
        Creates a `MeanRankMetric` instance.

        Args:
            name: string name of the metric instance.
            dtype: (Optional) data type of the metric result.
            pos: 2D tensor representing ranks/positions of records in a query
            mask: 2D tensor representing 0/1 mask for padded records

        NOTE: pos and mask should be same shape as y_pred and y_true
        """
        super(MeanRankMetric, self).__init__(self._compute, name, dtype=dtype, pos=pos, mask=mask)

    def _compute(self, y_true, y_pred, pos, mask):
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
        y_pred_click_ranks = tf.gather_nd(y_pred_ranks, indices=y_true_clicks)

        return self._get_matches_hook(y_pred_click_ranks)

    def _get_matches_hook(self, y_pred_click_ranks):
        raise NotImplementedError


class MRR(MeanRankMetric):
    """
    Custom metric class to compute Mean Reciprocal Rank


    """

    def __init__(self, name="MRR", **kwargs):
        super(MRR, self).__init__(name=name, **kwargs)

    def _get_matches_hook(self, y_pred_click_ranks):
        return math_ops.reciprocal(tf.cast(y_pred_click_ranks, tf.float32))


class ACR(MeanRankMetric):
    """
    Custom metric class to compute Average Click Rank


    """

    def __init__(self, name="ACR", **kwargs):
        super(ACR, self).__init__(name=name, **kwargs)

    def _get_matches_hook(self, y_pred_click_ranks):
        return tf.cast(y_pred_click_ranks, tf.float32)
