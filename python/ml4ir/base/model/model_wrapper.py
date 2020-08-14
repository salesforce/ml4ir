import tensorflow as tf
from tensorflow import keras

from ml4ir.base.model.losses.loss_base import RelevanceLossBase


class KerasModelWrapper(keras.Model):
    def __init__(self, inputs, outputs, output_name="score", **kwargs):
        self.output_name = output_name
        self.loss_metric = None
        super(KerasModelWrapper, self).__init__(inputs, outputs, **kwargs)

    def compile(self, **kwargs):
        # Define metric to track loss
        self.loss_metric = keras.metrics.Mean(name="loss")
        super(KerasModelWrapper, self).compile(**kwargs)

    def get_loss_value(self, y_true, y_pred, features):
        if isinstance(self.loss, str):
            loss_value = self.compiled_loss(y_true=y_true, y_pred=y_pred[self.output_name])
        elif isinstance(self.loss, RelevanceLossBase):
            loss_value = self.loss(
                y_true=y_true, y_pred=y_pred[self.output_name], features=features
            )
            # Update loss metric
            self.loss_metric.update_state(loss_value)
        elif isinstance(self.loss, keras.losses.Loss):
            loss_value = self.compiled_loss(y_true=y_true, y_pred=y_pred[self.output_name])
        else:
            raise KeyError("Unknown Loss encountered in KerasModelWrapper")

        return loss_value

    def train_step(self, data):
        X, y = data

        with tf.GradientTape() as tape:
            y_pred = self(X, training=True)
            loss_value = self.get_loss_value(y_true=y, y_pred=y_pred, features=X)

        # Compute gradients
        gradients = tape.gradient(loss_value, self.trainable_variables)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)
        # self.compiled_metrics.update_state(y, y_pred, features=X)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        X, y = data

        y_pred = self(X, training=False)

        # Update loss metric
        self.get_loss_value(y_true=y, y_pred=y_pred, features=X)

        # Update metrics
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

    @property
    def metrics(self):
        return [self.loss_metric] + super().metrics
