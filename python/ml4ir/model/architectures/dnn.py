import tensorflow as tf
from tensorflow.keras import layers
from typing import List


class DNNLayer:
    DENSE = "dense"
    BATCH_NORMALIZATION = "batch_norm"
    DROPOUT = "dropout"
    ACTIVATION = "activation"


class DNN:
    def __init__(self, model_config):
        self.layer_ops: List = self.define_architecture(model_config)

    def define_architecture(self, model_config):
        layer_ops = list()
        for layer_args in model_config["layers"]:
            layer_type = layer_args.pop("type")
            if layer_type == DNNLayer.DENSE:
                layer_op = layers.Dense(**layer_args)
            elif layer_type == DNNLayer.BATCH_NORMALIZATION:
                layer_op = layers.BatchNormalization(**layer_args)
            elif layer_type == DNNLayer.DROPOUT:
                layer_op = layers.Dropout(**layer_args)
            elif layer_type == DNNLayer.ACTIVATION:
                layer_op = layers.Activation(**layer_args)
            else:
                raise KeyError("Layer type is not supported : {}".format(layer_type))

            layer_ops.append(layer_op)

        return layer_ops

    def get_architecture_op(self):
        def _architecture_op(ranking_features):
            layer_input = ranking_features

            # Pass ranking features through all the layers of the DNN
            for layer_op in self.layer_ops:
                layer_input = layer_op(layer_input)

            # Collapse extra dimensions
            scores = tf.squeeze(layer_input, axis=-1)

            return scores

        return _architecture_op
