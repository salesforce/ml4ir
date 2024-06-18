import unittest
import numpy as np

from ml4ir.base.model.layers.normalization import Reciprocal, ArcTanNormalization

class TestReciprocal(unittest.TestCase):

    def test_reciprocal_default(self):
        reciprocal_op = Reciprocal()
        input_feature = np.array([[0.0, 0.8, 0.9, 0.5, 0.6, 0.7]])

        actual_reciprocals = reciprocal_op(input_feature).numpy()
        expected_reciprocals = np.array([[0., 1. / 0.8, 1. / 0.9, 1. / 0.5, 1. / 0.6, 1. / 0.7]])

        self.assertTrue(np.isclose(actual_reciprocals, expected_reciprocals).all())

    def test_reciprocal_2d_vs_3d(self):
        reciprocal_op = Reciprocal()
        input_feature = np.array([[0.0, 0.8, 0.9, 0.5, 0.6, 0.7]])

        actual_reciprocals_2d = reciprocal_op(input_feature).numpy()
        actual_reciprocals_3d = reciprocal_op(input_feature[:, :, np.newaxis]).numpy()

        self.assertTrue(np.isclose(np.squeeze(actual_reciprocals_2d), np.squeeze(actual_reciprocals_3d)).all())

    def test_reciprocal_k(self):
        reciprocal_op = Reciprocal(k=60)
        input_feature = np.array([[0.0, 0.8, 0.9, 0.5, 0.6, 0.7]])

        actual_reciprocals = reciprocal_op(input_feature).numpy()
        expected_reciprocals = np.array([[1. / (60 + 0.), 1. / (60 + 0.8), 1. / (60 + 0.9), 1. / (60 + 0.5), 1. / (60 + 0.6), 1. / (60 + 0.7)]])

        self.assertTrue(np.isclose(actual_reciprocals, expected_reciprocals).all())
    #
    def test_reciprocal_k_trainable(self):
        reciprocal_op = Reciprocal()
        self.assertEquals(len(reciprocal_op.trainable_variables), 0)
        self.assertEquals(len(reciprocal_op.non_trainable_variables), 1)

        reciprocal_op = Reciprocal(k_trainable=True)
        self.assertEquals(len(reciprocal_op.trainable_variables), 1)
        self.assertEquals(len(reciprocal_op.non_trainable_variables), 0)


class TestArcTanNormalization(unittest.TestCase):

    def test_arctan_norm_default(self):
        norm_op = ArcTanNormalization()
        input_feature = np.array([[0.0, 0.8, 0.9, 1.5, -0.6, 125.3]])

        actual_output = norm_op(input_feature).numpy()
        expected_output = [[0., 0.42955342, 0.46652454, 0.6256659, -0.34404173, 0.9949193]]

        self.assertTrue(np.isclose(actual_output, expected_output).all())
    #
    def test_arctan_norm_k(self):
        norm_op = ArcTanNormalization(k=0.1)
        input_feature = np.array([[0.0, 0.8, 0.9, 1.5, -0.6, 125.3]])

        actual_output = norm_op(input_feature).numpy()
        expected_output = [[0., 0.05082135, 0.05714183, 0.09478629, -0.03815145, 0.9492998]]

        self.assertTrue(np.isclose(actual_output, expected_output).all())
    #
    def test_arctan_norm_k_trainable(self):
        norm_op = ArcTanNormalization()
        self.assertEquals(len(norm_op.trainable_variables), 0)
        self.assertEquals(len(norm_op.non_trainable_variables), 1)

        norm_op = ArcTanNormalization(k_trainable=True)
        self.assertEquals(len(norm_op.trainable_variables), 1)
        self.assertEquals(len(norm_op.non_trainable_variables), 0)
