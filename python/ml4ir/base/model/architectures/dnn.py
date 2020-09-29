import tensorflow as tf
from tensorflow.keras import layers
from typing import List

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.features.feature_fns.categorical import get_vocabulary_info
from ml4ir.base.io.file_io import FileIO


OOV = 1



class DNNLayer:
    DENSE = "dense"
    BATCH_NORMALIZATION = "batch_norm"
    DROPOUT = "dropout"
    ACTIVATION = "activation"


class DNN:
    def __init__(self, model_config: dict, feature_config: FeatureConfig, file_io):
        self.file_io: FileIO = file_io
        self.layer_ops: List = self.define_architecture(model_config, feature_config)

    def define_architecture(self, model_config: dict, feature_config: FeatureConfig):
        """
        Convert the model from model_config to a List of tensorflow.keras.layer

        :param model_config: dict corresponding to the model config
        :param feature_config: dict corresponding to the feature config, only used in case of classification if the last
            layer of the model_config doesn't have a units number defined (or set to -1). In which case we retrieve the
            label vocabulary defined in the feature_config to deduce the number of units.
        :return: List[layers]: list of keras layer corresponding to each of the layers defined in the model_config.
        """
        def get_op(layer_type, layer_args):
            if layer_type == DNNLayer.DENSE:
                if not "units" in layer_args or layer_args["units"] == -1:
                    try:
                        label_feature_info = feature_config.get_label()
                        vocabulary_keys, vocabulary_ids = get_vocabulary_info(label_feature_info, self.file_io)
                        layer_args["units"] = len(vocabulary_keys) + OOV
                    except:
                        raise KeyError("We were not able to find information for the output layer of your DNN. "
                                       "Try specifying the number of output units either by passing \"units\" in the "
                                       "model configuration yaml file or units in the feature configuration file.")
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
            if isinstance(self.layer_ops[-1], layers.Dense) and (self.layer_ops[-1].units == 1):
                scores = tf.squeeze(layer_input, axis=-1)
            else:
                scores = layer_input

            return scores

        return _architecture_op
