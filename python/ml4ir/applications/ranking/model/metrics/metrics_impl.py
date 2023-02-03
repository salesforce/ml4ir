import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.python.ops import math_ops


class MeanRankMetric(metrics.Mean):
    """
    Mean metric for the ranks of a query
    """
    def update_state(self, y_true, y_pred, sample_weight=None):
        """
        Accumulates metric statistics by computing the mean of the
        metric function

        Parameters
        ----------
        y_true : Tensor object
            The ground truth values. Shape : [batch_size, max_sequence_size]
        y_pred : Tensor object
            The predicted values. Shape : [batch_size, max_sequence_size]
        sample_weight : Tensor object
            Optional weighting of each example. Defaults to 1. Can be
            a `Tensor` whose rank is either 0, or the same rank as `y_true`,
            and must be broadcastable to `y_true`.

        Returns
        -------
            Updated state of the metric

        Notes
        -----
        `y_true` and `y_pred` should have the same shape.
        """
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

        # Post processing on click ranks before mean
        query_scores = self._process_click_ranks(click_ranks)

        return super().update_state(query_scores, sample_weight=sample_weight)

    def _process_click_ranks(self, click_ranks):
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
    >>> then the MRR is 0.75
    """
    def _process_click_ranks(self, click_ranks):
        """
        Return reciprocal click ranks for MRR

        Parameters
        ----------
        click_ranks: Tensor object
            Tensor object containing the ranks of the clicked records for each query

        Returns
        -------
        Tensor object
            Reciprocal ranks tensor
        """
        return math_ops.reciprocal(tf.cast(click_ranks, tf.float32))


class ACR(MeanRankMetric):
    """
    Custom metric class to compute the Average Click Rank.

    Calculates the mean of the ranks of the
    clicked records from a list of queries.

    Examples
    --------
    >>> `y_true` is [[0, 0, 1], [0, 1, 0]]
    >>> `y_pred` is [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
    >>> then the ACR is 1.50
    """
    def _process_click_ranks(self, click_ranks):
        """
        Return click ranks for ACR

        Parameters
        ----------
        click_ranks: Tensor object
            Tensor object containing the ranks of the clicked records for each query

        Returns
        -------
        Tensor object
            Ranks tensor cast to float
        """
        return tf.cast(click_ranks, tf.float32)
