import unittest

import tensorflow as tf
from tensorflow.experimental.numpy import isclose
import numpy as np

from ml4ir.applications.ranking.model.layers.set_rank_encoder import SetRankEncoder


class TestSetRankEncoder(unittest.TestCase):
    """Unit tests for ml4ir.applications.ranking.model.layers.set_rank_encoder"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_features = 16
        self.encoding_size = 8

    def validate_encoder_output(self, encoder_output, mask):
        # Check that the encodings for masked inputs are all the same in a given query
        for i in range(mask.shape[0]):
            # Skip query if all documents are unmasked
            if tf.reduce_sum(mask[i]) == len(mask[i]):
                continue
            encoder_output_for_masked_i = tf.gather_nd(encoder_output[i], tf.where(mask[i] == 0))
            self.assertTrue(tf.reduce_all(tf.reduce_all(
                isclose(encoder_output_for_masked_i, encoder_output_for_masked_i[0, :]), axis=1)).numpy())

        # Check that the encodings for unmasked inputs are not all the same
        for i in range(mask.shape[0]):
            # Skip query if there is only 1 unmasked document
            if tf.reduce_sum(mask[i]) <= 1:
                continue
            encoder_output_i = tf.gather_nd(encoder_output[i], tf.where(mask[i] == 1))
            self.assertFalse(tf.reduce_all(tf.reduce_all(
                isclose(encoder_output_i, encoder_output_i[0, :]), axis=1)).numpy())

    def test_input_projection_op(self):
        """Test if the input_projection_op behaves as expected"""
        encoder = SetRankEncoder(self.encoding_size)

        x = np.random.randn(4, 5, self.num_features)
        projected_input = encoder.input_projection_op(x)

        self.assertEqual(x.shape[0], projected_input.shape[0])
        self.assertEqual(x.shape[1], projected_input.shape[1])
        self.assertNotEqual(x.shape[2], projected_input.shape[2])

        # Check if input feature dimension gets mapped to encoding dimension
        self.assertEqual(projected_input.shape[2], self.encoding_size)

    def test_transformer_encoder(self):
        """Test is the transformer_encoder behaves as expected"""
        encoder = SetRankEncoder(self.encoding_size)

        x = np.random.randn(4, 5, self.encoding_size)
        encoder_output = encoder.transformer_encoder(x)

        self.assertEqual(x.shape[0], encoder_output.shape[0])
        self.assertEqual(x.shape[1], encoder_output.shape[1])
        self.assertEqual(x.shape[2], encoder_output.shape[2])

        self.assertEqual(encoder_output.shape[2], self.encoding_size)

    def test_transformer_encoder_with_attention_mask(self):
        """Test is the transformer_encoder behaves as expected with attention mask"""
        encoder = SetRankEncoder(self.encoding_size)

        x = np.random.randn(4, 5, self.encoding_size)
        mask = np.random.binomial(n=1, p=0.5, size=[4 * 5]).reshape(4, 5)
        attention_mask = np.matmul(mask[:, :, np.newaxis], mask[:, np.newaxis, :])
        masked_x = (x.T * mask.T).T

        encoder_output = encoder.transformer_encoder(masked_x, attention_mask)

        self.assertEqual(x.shape[0], encoder_output.shape[0])
        self.assertEqual(x.shape[1], encoder_output.shape[1])
        self.assertEqual(x.shape[2], encoder_output.shape[2])

        self.assertEqual(encoder_output.shape[2], self.encoding_size)

        self.validate_encoder_output(encoder_output, mask)

    def test_transformer_encoder_kwargs(self):
        """Test if passing additional key-value args to TransformerEncoder works"""
        encoder = SetRankEncoder(self.encoding_size,
                                 num_layers=4,
                                 intermediate_size=32)
        config = encoder.get_config()

        # Default Args
        self.assertEqual(config["projection_dropout_rate"], 0.)
        self.assertEqual(config["num_attention_heads"], 8)

        # Passed Args
        self.assertEqual(config["encoding_size"], self.encoding_size)
        self.assertEqual(config["num_layers"], 4)
        self.assertEqual(config["intermediate_size"], 32)

    def test_set_rank_encoder_layer(self):
        """Test the SetRankEncoder call() function end-to-end"""
        encoder = SetRankEncoder(self.encoding_size,
                                 requires_mask=False)

        x = np.random.randn(4, 5, self.num_features)
        encoder_output = encoder(x)

        self.assertEqual(x.shape[0], encoder_output.shape[0])
        self.assertEqual(x.shape[1], encoder_output.shape[1])
        self.assertNotEqual(x.shape[2], encoder_output.shape[2])

        # Check if input feature dimension gets mapped to encoding dimension
        self.assertEqual(encoder_output.shape[2], self.encoding_size)

    def test_set_rank_encoder_layer_with_dropout(self):
        """Test the SetRankEncoder call() function end-to-end with dropout"""
        encoder = SetRankEncoder(self.encoding_size,
                                 requires_mask=False,
                                 dropout_rate=0.5)

        x = np.random.randn(4, 5, self.num_features)

        self.assertFalse(tf.reduce_all(encoder(x, training=True) == encoder(x, training=True)).numpy())
        self.assertFalse(tf.reduce_all(encoder(x, training=False) == encoder(x, training=True)).numpy())
        self.assertTrue(tf.reduce_all(encoder(x, training=False) == encoder(x, training=False)).numpy())

    def test_set_rank_encoder_layer_with_mask(self):
        """Test the SetRankEncoder call() function end-to-end with mask"""
        encoder = SetRankEncoder(self.encoding_size)

        x = np.random.randn(4, 5, self.num_features)
        mask = np.random.binomial(n=1, p=0.5, size=[4 * 5]).reshape(4, 5)

        encoder_output = encoder(x, mask)

        self.assertEqual(x.shape[0], encoder_output.shape[0])
        self.assertEqual(x.shape[1], encoder_output.shape[1])
        self.assertNotEqual(x.shape[2], encoder_output.shape[2])

        self.assertEqual(encoder_output.shape[2], self.encoding_size)

        self.validate_encoder_output(encoder_output, mask)
