from ml4ir.base.features.preprocessing import *
import unittest
import tensorflow as tf
import numpy as np
import warnings
warnings.filterwarnings("ignore")

class TestClickConversion(unittest.TestCase):
    def setUp(self):
        pass

    def test_1(self):
        typ = 'int'
        label_vector = np.ones(10, dtype=typ)
        label_vector = tf.convert_to_tensor(label_vector)
        clicks = convert_label_to_clicks(label_vector, typ)
        comp = tf.equal(label_vector, clicks)
        assert sum(tf.dtypes.cast(comp, 'int8')) == 10

        typ = 'float'
        label_vector = np.ones(10, dtype=typ)
        label_vector[0] = 5;
        label_vector[-1] = 5
        label_vector = tf.convert_to_tensor(label_vector)
        clicks = convert_label_to_clicks(label_vector, typ)
        assert clicks[0] == 1 and clicks[-1] == 1 and sum(clicks[1:-1]) == 0

        typ = 'int'
        label_vector = np.zeros(10, dtype=typ)
        label_vector = tf.convert_to_tensor(label_vector)
        clicks = convert_label_to_clicks(label_vector, typ)
        assert sum(tf.dtypes.cast(comp, 'int8')) == 10


    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
