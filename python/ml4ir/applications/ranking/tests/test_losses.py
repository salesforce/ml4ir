import os
import numpy as np
import tensorflow as tf

from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.applications.ranking.model.losses import pointwise_losses
from ml4ir.applications.ranking.model.losses import listwise_losses


class RankingModelTest(RankingTestBase):
    def setUp(self):

        super().setUp()

        self.logits = tf.constant(
            [[0.19686124, 0.73846658, 0.17136828, -0.11564828, -0.3011037],
             [-1.47852199, -0.71984421, -0.46063877, 1.05712223, 0.34361829],
             [-1.76304016, 0.32408397, -0.38508228, -0.676922, 0.61167629]])
        self.y_true = tf.constant(
            [[0., 0., 1., 0., 0.],
             [0., 1., 0., 0., 0.],
             [0., 1., 0., 0., 0.]])
        self.y_true_aux = tf.constant(
            [[0.34, 0.81, 0.22, -0.05, -0.67],
             [-1.35, -0.75, -0.37, 1.5467, 0.8],
             [-1.4356, 0.75, -0.6857, -0.689, 0.089]])
        self.y_true_aux_ties = tf.constant(
            [[0.34, 0.81, 0.22, 0.22, 0.22],
             [-1.35, -0.75, 1.5467, 1.5467, 0.8],
             [0, 0, 0, 0, 0]])
        self.mask = tf.constant(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 0.],
             [1., 1., 0., 0., 0.]])

    def test_sigmoid_cross_entropy(self):
        """Test the sigmoid cross entropy pointiwse loss object"""
        loss_op = pointwise_losses.SigmoidCrossEntropy()

        y_pred = loss_op.final_activation_op({
            "logits": self.logits,
            "metadata": {
                "mask": self.mask
            }
        })
        assert np.isclose(y_pred[0][0].numpy(), 0.54905695, atol=1e-5)
        assert np.isclose(y_pred[2][4].numpy(), 0., atol=1e-5)

        loss = loss_op({"mask": self.mask}, self.y_true, y_pred)
        assert np.isclose(loss, 0.6905699, atol=1e-5)

    def test_softmax_cross_entropy(self):
        """Test the softmax cross entropy listwise loss object"""
        loss_op = listwise_losses.SoftmaxCrossEntropy()

        y_pred = loss_op.final_activation_op({
            "logits": self.logits,
            "metadata": {
                "mask": self.mask
            }
        })
        assert np.isclose(y_pred[0][0].numpy(), 0.19868991, atol=1e-5)
        assert np.isclose(y_pred[2][4].numpy(), 0.0, atol=1e-5)

        loss = loss_op({"mask": self.mask}, self.y_true, y_pred)
        assert np.isclose(loss, 1.306335, atol=1e-5)

    def test_rank_one_list_net(self):
        """Test the rank-one listnet listwise loss object"""
        loss_op = listwise_losses.RankOneListNet()

        y_pred = loss_op.final_activation_op({
            "logits": self.logits,
            "metadata": {
                "mask": self.mask
            }
        })

        assert np.isclose(y_pred[0][0].numpy(), 0.19868991, atol=1e-5)
        assert np.isclose(y_pred[2][4].numpy(), 0.0, atol=1e-5)

        loss = loss_op({"mask": self.mask}, self.y_true, y_pred)
        assert np.isclose(loss, 2.1073625, atol=1e-5)

    def test_aux_one_hot_hot_cross_entropy(self):
        """Test the auxiliary one hot cross entropy listwise loss object"""
        loss_op = listwise_losses.AuxiliaryOneHotCrossEntropy()

        y_pred = loss_op.final_activation_op({
            "logits": self.logits,
            "metadata": {
                "mask": self.mask
            }
        })

        assert np.isclose(y_pred[0][0].numpy(), 0.19868991, atol=1e-5)
        assert np.isclose(y_pred[2][4].numpy(), 0.0, atol=1e-5)

        loss = loss_op({"mask": self.mask}, self.y_true_aux, y_pred)
        assert np.isclose(loss, 0.5249801, atol=1e-5)

    def test_aux_one_hot_cross_entropy_with_ties(self):
        """Test the one hot cross entropy for aux target with ties"""
        loss_op = listwise_losses.AuxiliaryOneHotCrossEntropy()

        y_pred = loss_op.final_activation_op({
            "logits": self.logits,
            "metadata": {
                "mask": self.mask
            }
        })

        assert np.isclose(y_pred[0][0].numpy(), 0.19868991, atol=1e-5)
        assert np.isclose(y_pred[2][4].numpy(), 0.0, atol=1e-5)

        loss = loss_op({"mask": self.mask}, self.y_true_aux_ties, y_pred)
        assert np.isclose(loss, 0.8936954, atol=1e-5)

    def test_aux_softmax_cross_entropy(self):
        """Test the auxiliary softmax cross entropy listwise loss object"""
        loss_op = listwise_losses.AuxiliarySoftmaxCrossEntropy()

        y_pred = loss_op.final_activation_op({
            "logits": self.logits,
            "metadata": {
                "mask": self.mask
            }
        })

        assert np.isclose(y_pred[0][0].numpy(), 0.19868991, atol=1e-5)
        assert np.isclose(y_pred[2][4].numpy(), 0.0, atol=1e-5)

        loss = loss_op({"mask": self.mask}, self.y_true_aux, y_pred)
        assert np.isclose(loss, 0.88127804, atol=1e-5)