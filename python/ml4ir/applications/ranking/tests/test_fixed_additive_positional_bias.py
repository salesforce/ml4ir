import unittest
import warnings
import tensorflow as tf
from ml4ir.base.model.layers.fixed_additive_positional_bias import FixedAdditivePositionalBias

warnings.filterwarnings("ignore")

INPUT_DIR = "/ml4ir/applications/ranking/tests/data/configs"

class TestFixedAdditivePositionalBias(unittest.TestCase):

    def apply_additive_positional_bias(self, rank_index, max_ranks, training):
        """Tests if the positional bias is applied as expected during training (training=True) or evaluation."""
        positional_bias = FixedAdditivePositionalBias(max_ranks)
        biases = positional_bias(tf.convert_to_tensor(rank_index), training)
        if not training:
            assert tf.math.reduce_sum(biases).numpy() == 0.0
        else:
            for i in range(len(rank_index)):
                assert biases[i][0].numpy() == positional_bias.weights[0][rank_index[i] - 1].numpy()[0]

    def test_one_hot_conversion(self):
        """calling additive positional bias"""
        self.apply_additive_positional_bias([5, 4], 5, True)
        self.apply_additive_positional_bias([1,3], 5, False)
        self.apply_additive_positional_bias([1,2], 2, True)
        self.apply_additive_positional_bias([2], 2, False)
        self.apply_additive_positional_bias([2,4,6,8], 10, True)

    def test_zeros_weight_initialization(self):
        """Testing weight initializations to zeros"""
        positional_bias = FixedAdditivePositionalBias(max_ranks=5, kernel_initializer='Zeros')
        positional_bias(tf.constant([1.]))
        weights = positional_bias.dense.get_weights()
        assert all([w == 0. for w in weights[0]])

    def test_non_zeros_weight_initialization(self):
        """Testing weight initializations to non zeros"""
        positional_bias = FixedAdditivePositionalBias(max_ranks=5, kernel_initializer='glorot_uniform')
        positional_bias(tf.constant([1.]))
        weights = positional_bias.dense.get_weights()
        assert any([w != 0. for w in weights[0]])



if __name__ == "__main__":
    unittest.main()