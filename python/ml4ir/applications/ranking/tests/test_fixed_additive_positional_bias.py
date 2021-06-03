import unittest
import warnings
import tensorflow as tf
from ml4ir.base.model.architectures.fixed_additive_positional_bias import FixedAdditivePositionalBias
import numpy as np

warnings.filterwarnings("ignore")


class TestFixedAdditivePositionalBias(unittest.TestCase):
    def setUp(self):
        self.positional_bias = FixedAdditivePositionalBias()

    def one_hot_conversion(self, rank_index, max_ranks, training):
        """Convert to a one-hot tensor"""
        one_hot = self.positional_bias.convert_to_one_hot(tf.convert_to_tensor(rank_index), max_ranks, training)
        for i in range(len(rank_index)):
            expected = np.zeros(max_ranks)
            if training:
                expected[rank_index[i] - 1] = 1
            assert (one_hot[i].numpy() == expected).all()

    def test_one_hot_conversion(self):
        self.one_hot_conversion([1,3], 5, False)
        self.one_hot_conversion([5,4], 5, True)
        self.one_hot_conversion([1,2], 2, True)
        self.one_hot_conversion([2], 2, False)
        self.one_hot_conversion([2,4,6,8], 10, True)


if __name__ == "__main__":
    unittest.main()
