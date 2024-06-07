import tensorflow as tf
from tensorflow.keras import metrics


class SegmentMean(metrics.Metric):
    """
    Keras Metrics class that computes a mean of the segments instead of overall mean.

    Can be used to compute macro, micro averages by overriding result()
    """

    def __init__(self, num_segments: int, name: str = "segment_mean", **kwargs):
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
        self.num_segments = tf.constant(num_segments)
        self.total_sum = self.add_weight(name="total_sum", shape=(num_segments,),
                                         initializer="zeros")
        self.total_count = self.add_weight(name="total_count", shape=(num_segments,),
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
        segment_sum = tf.math.unsorted_segment_sum(values, segments, num_segments=self.num_segments)
        segment_count = tf.math.unsorted_segment_sum(tf.ones_like(values), segments,
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
        return tf.math.divide_no_nan(self.total_sum, self.total_count)
