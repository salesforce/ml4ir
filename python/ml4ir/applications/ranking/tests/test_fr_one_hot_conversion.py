import unittest
import warnings
from ml4ir.base.features.preprocessing import convert_fr_to_one_hot
import numpy as np

warnings.filterwarnings("ignore")


class TestFrOneHotConversion(unittest.TestCase):

    def fr_one_hot_conversion(self, fr_value, max_ranks, mask_fr):
        """Convert to a one-hot vector"""
        one_hot = convert_fr_to_one_hot(fr_value, max_ranks, mask_fr)
        expected = np.zeros(max_ranks)
        if not mask_fr:
            expected[fr_value - 1] = 1
        assert (one_hot.numpy() == expected).all()

    def test_fr_one_hot_conversion(self):
        self.fr_one_hot_conversion([1], 25, False)
        self.fr_one_hot_conversion([5], 25, True)
        self.fr_one_hot_conversion(1, 5, False)
        self.fr_one_hot_conversion(5, 5, True)
        self.fr_one_hot_conversion(1, 2, True)
        self.fr_one_hot_conversion(2, 2, False)


if __name__ == "__main__":
    unittest.main()
