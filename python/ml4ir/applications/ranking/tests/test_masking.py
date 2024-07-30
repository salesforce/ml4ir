import unittest
import numpy as np
import tensorflow as tf

from ml4ir.applications.ranking.model.layers.masking import RecordFeatureMask, QueryFeatureMask


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


class TestQueryFeatureMask(unittest.TestCase):
    """Tests for ml4ir.applications.ranking.model.layers.masking.RecordFeatureMask"""

    def test_mask_rate(self):
        """Test masking occurs at the mask_rate"""
        mask_rate = 0.5
        mask_op = QueryFeatureMask(mask_rate=mask_rate)
        input = np.ones((10, 5, 1))

        num_iter = 10000
        masked_sum = 0.
        for i in range(num_iter):
            masked_sum += mask_op(input, training=True).numpy().sum()
        actual_mask_rate = 1. - (masked_sum / (num_iter * input.sum()))

        self.assertTrue(np.isclose(actual_mask_rate, mask_rate, atol=1e-2))

    def test_mask_all_query_feature(self):
        mask_rate = 0.5
        mask_op = QueryFeatureMask(mask_rate=mask_rate)
        input = np.ones((50, 5, 1))
        input = mask_op(input, training=True)

        # Sum along sequence_len dimension
        sum_along_sequence = tf.reduce_sum(input, axis=1)

        # Sum along feature_dim to get the total sum for each batch
        total_sum_per_batch = tf.reduce_sum(sum_along_sequence, axis=1)

        # Check if each sum is either 5 or 0 (i.e. all records per query are masked or not)
        condition = tf.reduce_all(tf.logical_or(tf.equal(total_sum_per_batch, 5), tf.equal(total_sum_per_batch, 0)))
        self.assertTrue(condition.numpy(), True)

    def test_create_fixed_masks(self):
        mask = QueryFeatureMask(name="query_feature_mask",
                                mask_rate=0,
                                mask_at_inference=True,
                                requires_mask=True)
        feature_dim = 2
        expected_result = [(0, 0), (0, 1), (1, 0), (1, 1)]
        result = mask.create_fixed_masks(feature_dim)
        self.assertEqual(result, expected_result)

        feature_dim = 3
        expected_result = [
            (0, 0, 0), (0, 0, 1), (0, 1, 0), (0, 1, 1),
            (1, 0, 0), (1, 0, 1), (1, 1, 0), (1, 1, 1)
        ]
        result = mask.create_fixed_masks(feature_dim)
        self.assertEqual(result, expected_result)

    def test_apply_fixed_mask(self):
        mask = QueryFeatureMask(name="query_feature_mask",
                                mask_rate=0,
                                mask_at_inference=True,
                                requires_mask=True)

        inputs = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
        mask.create_fixed_masks = lambda x: [(1, 0), (0, 1)]

        # Run first time, expecting first mask (1, 0)
        result = mask.apply_fixed_mask(inputs, training=True)
        expected_result = tf.constant([[[0., 2.], [0., 4.]], [[0., 6.], [0., 8.]]], dtype=tf.float32)
        tf.debugging.assert_near(result, expected_result)

        # Run second time, expecting second mask (0, 1)
        result = mask.apply_fixed_mask(inputs, training=True)
        expected_result = tf.constant([[[1., 0.], [3., 0.]], [[5., 0.], [7., 0.]]], dtype=tf.float32)
        tf.debugging.assert_near(result, expected_result)

    def test_apply_stochastic_mask_training(self):
        mask = QueryFeatureMask(name="query_feature_mask",
                                mask_rate=0,
                                mask_at_inference=True,
                                requires_mask=True)
        inputs = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], dtype=tf.float32)
        mask.mask_rate = 0.9
        result1 = mask.apply_stochastic_mask(inputs, training=True)
        tf.debugging.assert_none_equal(result1, inputs)

        mask.mask_rate = 0
        result2 = mask.apply_stochastic_mask(inputs, training=True)
        tf.debugging.assert_equal(result2, inputs)

