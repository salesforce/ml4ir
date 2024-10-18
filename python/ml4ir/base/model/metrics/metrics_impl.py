import tensorflow as tf
from tensorflow.keras import metrics

from typing import List


class SegmentMean(metrics.Metric):
    """
    Keras Metrics class that computes a mean of the segments instead of overall mean.

    Can be used to compute macro, micro averages by overriding result()
    """

    def __init__(self, segments: List[str] = [], name: str = "segment_mean", **kwargs):
        """
        Instantiate a SegmentMean object

        Parameters
        ----------
        num_segments: int
            Number of segments to be used for computing the mean
        name: str
            Name of the metric
        **kwargs: dict
            Additional keyword arguments to be passed to the metric constructor
        """
        super().__init__(name=name, **kwargs)
        if not segments:
            raise ValueError(f"Invalid argument passed for segments -> {segments}")
        self.segments = segments
        # NOTE: number of segments is vocabulary + 1 for OOV bucket
        self.num_segments = len(segments) + 1
        self.segment_lookup = tf.keras.layers.StringLookup(vocabulary=segments,
                                                           num_oov_indices=1)
        self.total_sum = self.add_weight(name="total_sum", shape=(self.num_segments,),
                                         initializer="zeros")
        self.total_count = self.add_weight(name="total_count", shape=(self.num_segments,),
                                           initializer="zeros")

    def reset_state(self):
        """Reset the state of the metric to initial configuration"""
        tf.keras.backend.batch_set_value([(v, tf.zeros((self.num_segments,))) for v in self.variables])

    def update_state(self, values, segments, sample_weight=None):
        """
        Accumulates metric statistics from input to update state variables of the metric

        Parameters
        ----------
        values: tf.Tensor
            Tensor object that contains the values to be used for mean
        segments: tf.Tensor
            Tensor object that contains the segment identifiers for each value
        sample_weight: tf.Tensor
            Optional weighting of each example
        """
        segment_ids = self.segment_lookup(segments)
        segment_sum = tf.math.unsorted_segment_sum(values, segment_ids, num_segments=self.num_segments)
        segment_count = tf.math.unsorted_segment_sum(tf.ones_like(values), segment_ids,
                                                     num_segments=self.num_segments)
        self.total_sum.assign_add(segment_sum)
        self.total_count.assign_add(segment_count)

    def result(self):
        """
        Compute the metric value for the current state variables

        Returns
        -------
        tf.Tensor
            Tensor object with the mean for each segment
            Shape -> (num_segments,)
        """
        # Compute means for each segment and exclude OOV segment ID=0
        return tf.math.divide_no_nan(self.total_sum, self.total_count)[1:]
