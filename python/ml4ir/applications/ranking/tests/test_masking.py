import unittest
import numpy as np

from ml4ir.applications.ranking.model.layers.masking import RecordFeatureMask


class TestRecordFeatureMask(unittest.TestCase):
    """Tests for ml4ir.applications.ranking.model.layers.masking.RecordFeatureMask"""

    def test_mask_rate(self):
        """Test masking occurs at the mask_rate"""
        mask_rate = 0.5
        mask_op = RecordFeatureMask(mask_rate=mask_rate)
        input = np.ones((2, 5, 4))

        num_iter = 10000
        masked_sum = 0.
        for i in range(num_iter):
            masked_sum += mask_op(input, training=True).numpy().sum()
        actual_mask_rate = 1. - (masked_sum / (num_iter * input.sum()))

        self.assertTrue(np.isclose(actual_mask_rate, mask_rate, atol=1e-2))

    def test_mask_all_features(self):
        """Test if all record's features are masked"""
        mask_op = RecordFeatureMask(mask_rate=0.5)
        feature_dim = 4
        masked_input = mask_op(np.ones((2, 5, feature_dim)), training=True)
        masked_record_sum = masked_input.numpy().sum(axis=-1)

        self.assertTrue(((masked_record_sum == 0.) | (masked_record_sum == feature_dim)).all())

    def test_no_masking_at_inference(self):
        """Test that masking does not happen at inference time"""
        mask_op = RecordFeatureMask(mask_rate=0.5)
        input = np.ones((2, 5, 4))

        inference_mask = mask_op(input)
        training_mask = mask_op(input, training=True)
        self.assertFalse(np.isclose(inference_mask, training_mask).all())
        self.assertTrue(inference_mask.numpy().sum() == (2 * 5 * 4))

    def test_masking_at_inference(self):
        """Test that masking happens at inference time"""
        mask_op = RecordFeatureMask(mask_rate=0.5, mask_at_inference=True)
        input = np.ones((2, 5, 4))

        inference_mask = mask_op(input)
        self.assertFalse(inference_mask.numpy().sum() == (2 * 5 * 4))