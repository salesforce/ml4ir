import unittest
import numpy as np

from ml4ir.base.model.metrics.metrics_impl import SegmentMean

class SegmentMeanTest(unittest.TestCase):

    def test_segment_mean_update_state(self):
        """Test that the mean of each segment is computed as expected"""
        segment_mean = SegmentMean(num_segments=3)

        with self.subTest("Check that the segment mean is zero"):
            self.assertTrue(np.isclose(segment_mean.result().numpy(), [0, 0, 0]).all())

        segment_mean.update_state(values=[1., 2., 3.], segments=[1, 0, 1])
        with self.subTest("Test if metric is updated after one computation"):
            self.assertTrue(np.isclose(segment_mean.result().numpy(), [2, 2, 0]).all())

        segment_mean.update_state(values=[5., 2.], segments=[2, 1])
        with self.subTest("Test if metric is updated after second computation"):
            self.assertTrue(np.isclose(segment_mean.result().numpy(), [2, 2, 5]).all())

    def test_segment_mean_reset_state(self):
        """Test that the variables are reset"""
        segment_mean = SegmentMean(num_segments=3)

        segment_mean.update_state(values=[1., 2., 3.], segments=[1, 0, 1])
        with self.subTest("Test that metric is non-zero after computation"):
            self.assertFalse(np.isclose(segment_mean.result().numpy(), [0, 0, 0]).all())

        segment_mean.reset_state()
        with self.subTest("Test that metric is zero after reset"):
            self.assertTrue(np.isclose(segment_mean.result().numpy(), [0, 0, 0]).all())