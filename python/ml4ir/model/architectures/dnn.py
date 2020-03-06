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
    def _get_dense_layer(**kwargs):
        return layers.Dense(**kwargs)

    @staticmethod
    def _get_batch_norm_layer(**kwargs):
        return layers.BatchNormalization(**kwargs)

    @staticmethod
    def _get_dropout_layer(**kwargs):
        return layers.Dropout(**kwargs)

    def define_architecture(self, model_config):
        layer_ops = list()
        for layer_info in model_config["layers"]:
            layer_type = layer_info.pop("type")
            if layer_type == DNNLayer.DENSE:
                layer_op = DNN._get_dense_layer(**layer_info)
            elif layer_type == DNNLayer.BATCH_NORMALIZATION:
                layer_op = DNN._get_batch_norm_layer(**layer_info)
            elif layer_type == DNNLayer.DROPOUT:
                layer_op = DNN._get_dropout_layer(**layer_info)
            else:
                raise KeyError("Dense layer type is not supported : {}".format(layer_type))

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
