import unittest
import warnings
import tensorflow as tf
from ml4ir.base.model.architectures.fixed_additive_positional_bias import FixedAdditivePositionalBias

warnings.filterwarnings("ignore")


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


if __name__ == "__main__":
    unittest.main()
