import tensorflow as tf
from tensorflow.keras import layers
from typing import List


class DNNLayer:
    DENSE = "dense"
    BATCH_NORMALIZATION = "batch_norm"
    DROPOUT = "dropout"


class DNN:
    def __init__(self, model_config):
        self.layer_ops: List = self.define_architecture(model_config)

    @staticmethod
    def _get_dense_layer(name, hidden_units, activation):
        return layers.Dense(int(hidden_units), activation=activation, name=name)

    @staticmethod
    def _get_batch_norm_layer(name):
        return layers.BatchNormalization(name=name)

    @staticmethod
    def _get_dropout_layer(name, rate):
        return layers.Dropout(rate=float(rate), name=name)

    def define_architecture(self, model_config):
        layer_ops = list()
        for layer in model_config["layers"]:
            if layer["type"] == DNNLayer.DENSE:
                layer_op = DNN._get_dense_layer(
                    name=layer["name"],
                    hidden_units=layer["hidden_units"],
                    activation=layer["activation"],
                )
            elif layer["type"] == DNNLayer.BATCH_NORMALIZATION:
                layer_op = DNN._get_batch_norm_layer(name=layer["name"])
            elif layer["type"] == DNNLayer.DROPOUT:
                layer_op = DNN._get_dropout_layer(name=layer["name"], rate=layer["rate"])
            else:
                raise KeyError("Dense layer type is not supported : {}".format(layer["type"]))

            layer_ops.append(layer_op)

        return layer_ops

    def get_architecture_op(self):
        def _architecture_op(ranking_features):
            _ = ranking_features

            # Pass ranking features through all the layers of the DNN
            for layer_op in self.layer_ops:
                _ = layer_op(_)

            # Collapse extra dimensions
            scores = tf.squeeze(_, axis=-1)

            return scores

        return _architecture_op
