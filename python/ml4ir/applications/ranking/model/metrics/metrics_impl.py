import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.python.ops import math_ops

from ml4ir.base.model.metrics.metrics_impl import SegmentMean


class ClickRankProcessor:
    """
    Class with click rank processing utilities
    """

    @staticmethod
    def get_click_ranks(y_true, y_pred, mask):
        """
        Gets the ranks of the clicked result for each query

        Parameters
        ----------
        y_true : Tensor object
            The ground truth values. Shape : [batch_size, max_sequence_size]
        y_pred : Tensor object
            The predicted values. Shape : [batch_size, max_sequence_size]
        mask : Tensor object
            Tensor object that contains 0/1 flag to identify which
            results were padded and thus should be excluded from metric computation

        Returns
        -------
            Ranks of the clicked results for each query
            Ties are broken with minimum ranks
        """
        # Convert predicted ranking scores into ranks for each record per query
        y_pred_ranks = tf.cast(
            tf.add(
                tf.argsort(
                    tf.argsort(y_pred, axis=-1, direction="DESCENDING", stable=True), stable=True
                ),
                tf.constant(1),
            ),
            tf.float32
        )

        # Fetch highest relevance grade from the y_true labels
        y_true_clicks = tf.cast(
            tf.equal(y_true, tf.math.reduce_max(y_true, axis=-1)[:, tf.newaxis]), tf.float32)

        # Compute rank of clicked record from predictions
        # Break ties in case of multiple target click labels using the min rank from y_pred_ranks
        click_ranks = tf.reduce_min(tf.divide(y_pred_ranks, y_true_clicks), axis=-1)

        return tf.cast(click_ranks, tf.float32)

    @staticmethod
    def process_click_ranks(click_ranks):
        raise NotImplementedError


class MeanRankMetric(metrics.Mean, ClickRankProcessor):
    """
    Mean metric for the ranks of a query
    """

    def update_state(self, y_true, y_pred, mask=None, sample_weight=None):
        """
        Accumulates metric statistics by computing the mean of the
        metric function

        Parameters
        ----------
        y_true : Tensor object
            The ground truth values. Shape : [batch_size, max_sequence_size]
        y_pred : Tensor object
            The predicted values. Shape : [batch_size, max_sequence_size]
        mask : Tensor object
            Tensor object that contains 0/1 flag to identify which
            results were padded and thus should be excluded from metric computation
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
        click_ranks = self.get_click_ranks(y_true, y_pred, mask)

        # Post processing on click ranks before mean
        query_scores = self.process_click_ranks(click_ranks)

        return super().update_state(query_scores, sample_weight=sample_weight)


class SegmentMeanRankMetric(SegmentMean, ClickRankProcessor):
    """
    Mean metric for the ranks of a query
    """

    def update_state(self, y_true, y_pred, segments, mask=None, sample_weight=None):
        """
        Accumulates metric statistics by computing the mean of the
        metric function

        Parameters
        ----------
        y_true : Tensor object
            The ground truth values. Shape : [batch_size, max_sequence_size]
        y_pred : Tensor object
            The predicted values. Shape : [batch_size, max_sequence_size]
        segments : Tensor object
            The segment identifiers for each query. Shape : [batch_size]
        mask : Tensor object
            Tensor object that contains 0/1 flag to identify which
            results were padded and thus should be excluded from metric computation
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
        click_ranks = self.get_click_ranks(y_true, y_pred, mask)

        # Post processing on click ranks before mean
        query_scores = self.process_click_ranks(click_ranks)

        return super().update_state(query_scores, segments=segments, sample_weight=sample_weight)


class MacroMeanRankMetric(SegmentMeanRankMetric):
    """
    Mean metric for the ranks of a query
    """

    def result(self):
        """
        Compute the metric value for the current state variables

        Returns
        -------
        tf.Tensor
            Tensor object with the mean for each segment
            Shape -> (num_segments,)
        """
        segment_means = super().result()
        return tf.math.reduce_mean(segment_means)


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

    @staticmethod
    def process_click_ranks(click_ranks):
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
        return math_ops.reciprocal(click_ranks)


class SegmentMRR(SegmentMeanRankMetric):
    """
    Custom metric class to compute the Mean Reciprocal Rank at the segment level.

    Calculates the mean of the reciprocal ranks of the
    clicked records from a list of queries.

    Examples
    --------
    >>> `y_true` is [[0, 0, 1], [0, 1, 0], [0, 1, 0]]
    >>> `y_pred` is [[0.1, 0.9, 0.8], [0.05, 0.95, 0], [0.05, 0.95, 0]]
    >>> `segments` is [0, 0, 1]
    >>> then the SegmentMRR is [0.75, 1.]
    """

    @staticmethod
    def process_click_ranks(click_ranks):
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
        return math_ops.reciprocal(click_ranks)


class MacroMRR(MacroMeanRankMetric):
    """
    Custom metric class to compute the Mean Reciprocal Rank macro averaged at the segment level.

    Calculates the mean of the reciprocal ranks of the
    clicked records from a list of queries.

    Examples
    --------
    >>> `y_true` is [[0, 0, 1], [0, 1, 0], [0, 1, 0]]
    >>> `y_pred` is [[0.1, 0.9, 0.8], [0.05, 0.95, 0], [0.05, 0.95, 0]]
    >>> `segments` is [0, 0, 1]
    >>> then the SegmentMRR is 0.875
    """

    def process_click_ranks(self, click_ranks):
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
        return math_ops.reciprocal(click_ranks)


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

    def process_click_ranks(self, click_ranks):
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
        return click_ranks


class NDCG(metrics.Mean):
    """
    Custom metric class for computing the Normalized Discounted Cumulative Gain (NDCG).

    Inherits from tf.keras.metrics.Mean.
    """

    def update_state(self, y_true, y_pred, mask=None, sample_weight=None):
        """
        Update the metric state with new inputs.

        Parameters:
            y_true (tf.Tensor): The tensor containing the true relevance scores.
            y_pred (tf.Tensor): The tensor containing the predicted scores.
            mask (tf.Tensor, optional): The tensor containing the mask values. Default is None.
            sample_weight (tf.Tensor, optional): The tensor containing the sample weights. Default is None.
        """
        y_true_masked = y_true
        y_pred_masked = y_pred
        if mask is not None:
            mask_float = tf.cast(mask, dtype=tf.float32)  # Convert mask to float32

            # Apply mask to the true relevance scores and predicted scores
            y_true_masked = y_true * mask_float
            y_pred_masked = y_pred * mask_float

        # Sort the predicted scores in descending order
        sorted_indices = tf.argsort(y_pred_masked, axis=-1, direction="DESCENDING")
        sorted_labels = tf.gather(y_true_masked, sorted_indices, axis=-1,
                                  batch_dims=len(y_true_masked.shape) - 1)

        # Compute the Discounted Cumulative Gain (DCG)
        positions = tf.range(1, tf.shape(y_true_masked)[-1] + 1, dtype=tf.float32)
        discounts = tf.math.log1p(positions) / tf.math.log1p(2.0)  # log base 2
        dcg = tf.reduce_sum(sorted_labels / discounts, axis=-1)

        # Sort the true relevance scores in descending order
        true_sorted_indices = tf.argsort(y_true_masked, axis=-1, direction="DESCENDING")
        true_sorted_labels = tf.gather(y_true_masked, true_sorted_indices, axis=-1,
                                       batch_dims=len(y_true_masked.shape) - 1)

        # Compute the Ideal Discounted Cumulative Gain (IDCG)
        idcg = tf.reduce_sum(true_sorted_labels / discounts, axis=-1)

        # Compute the Normalized Discounted Cumulative Gain (NDCG)
        ndcg = dcg / idcg

        # removing nans coming from having no relevance signals in y_true
        ndcg = tf.where(tf.math.is_nan(ndcg), tf.zeros_like(ndcg), ndcg)

        return super().update_state(ndcg, sample_weight=sample_weight)

    def _process_click_ranks(self, click_ranks):
        """
        Process the click ranks.

        Parameters:
            click_ranks (tf.Tensor): The tensor containing the click ranks.

        Returns:
            tf.Tensor: The processed click ranks.
        """
        return tf.cast(click_ranks, tf.float32)
