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
        def get_op(layer_type, layer_args):
            if layer_type == DNNLayer.DENSE:
                return layers.Dense(**layer_args)
            elif layer_type == DNNLayer.BATCH_NORMALIZATION:
                return layers.BatchNormalization(**layer_args)
            elif layer_type == DNNLayer.DROPOUT:
                return layers.Dropout(**layer_args)
            elif layer_type == DNNLayer.ACTIVATION:
                return layers.Activation(**layer_args)
            else:
                raise KeyError("Layer type is not supported : {}".format(layer_type))

        return [
            get_op(layer_args["type"], {k: v for k, v in layer_args.items() if k not in "type"})
            for layer_args in model_config["layers"]
        ]

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
