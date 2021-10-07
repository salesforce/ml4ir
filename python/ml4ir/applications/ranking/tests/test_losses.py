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
        self.mask = tf.constant(
            [[1., 1., 1., 1., 1.],
             [1., 1., 1., 1., 0.],
             [1., 1., 0., 0., 0.]])

    def test_sigmoid_cross_entropy(self):
        """Test the sigmoid cross entropy pointiwse loss object"""
        loss = pointwise_losses.SigmoidCrossEntropy()
        activation_op = loss.get_final_activation_op(output_name="y_pred")
        loss_fn = loss.get_loss_fn(mask=self.mask)

        y_pred = activation_op(logits=self.logits, mask=self.mask)
        assert np.isclose(y_pred[0][0].numpy(), 0.54905695, atol=1e-5)
        assert np.isclose(y_pred[2][4].numpy(), 0.64832306, atol=1e-5)

        assert np.isclose(loss_fn(self.y_true, y_pred), 0.6905699, atol=1e-5)

    def test_softmax_cross_entropy(self):
        """Test the softmax cross entropy listwise loss object"""
        loss = listwise_losses.SoftmaxCrossEntropy()
        activation_op = loss.get_final_activation_op(output_name="y_pred")
        loss_fn = loss.get_loss_fn(mask=self.mask)

        y_pred = activation_op(logits=self.logits, mask=self.mask)

        assert np.isclose(y_pred[0][0].numpy(), 0.19868991, atol=1e-5)
        assert np.isclose(y_pred[2][4].numpy(), 0., atol=1e-5)

        assert np.isclose(loss_fn(self.y_true, y_pred), 1.306335, atol=1e-5)

    def test_rank_one_list_net(self):
        """Test the rank-one listnet listwise loss object"""
        loss = listwise_losses.RankOneListNet()
        activation_op = loss.get_final_activation_op(output_name="y_pred")
        loss_fn = loss.get_loss_fn(mask=self.mask)

        y_pred = activation_op(logits=self.logits, mask=self.mask)

        assert np.isclose(y_pred[0][0].numpy(), 0.19868991, atol=1e-5)
        assert np.isclose(y_pred[2][4].numpy(), 0., atol=1e-5)

        assert np.isclose(loss_fn(self.y_true, y_pred), 2.1073625, atol=1e-5)
