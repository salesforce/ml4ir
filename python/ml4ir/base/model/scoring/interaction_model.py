import tensorflow as tf
from tensorflow import keras

from ml4ir.base.config.keys import FeatureTypeKey, TFRecordTypeKey, SequenceExampleTypeKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.features.feature_layer import FeatureLayerMap
from ml4ir.base.io.file_io import FileIO

from typing import Dict


FN = "fn"
NODE_NAME = "node_name"
NAME = "name"
MASK = "mask"
FEATURE_LAYER_INFO = "feature_layer_info"
TRAINABLE = "trainable"
TFRECORD_TYPE = "tfrecord_type"
DTYPE = "dtype"


class InteractionModel(keras.Model):
    """
    InteractionModel class that defines tensorflow layers that act on input features to
    convert them into numeric features to be fed into further neural network layers
    """

    def __init__(
        self,
        feature_config: FeatureConfig,
        tfrecord_type: str,
        feature_layer_keys_to_fns: dict = {},
        max_sequence_size: int = 0,
        file_io: FileIO = None,
        **kwargs
    ):
        """
        Constructor for instantiating a base InteractionModel

        Parameters
        ----------
        feature_config : `FeatureConfig` object
            FeatureConfig object that defines list of model features
            and the feature transformation functions to be used on each
        tfrecord_type : {"example", "sequence_example"}
            Type of TFRecord protobuf being used for model training
        feature_layer_keys_to_fns : dict
            Dictionary of custom feature transformation functions to be applied
            on the input features
        max_sequence_size : int, optional
            Maximum size of the sequence in SequenceExample protobuf
        file_io : FileIO object
            `FileIO` object that handles read write operations
        """
        super().__init__(**kwargs)

        self.feature_config = feature_config
        self.tfrecord_type = tfrecord_type
        self.feature_layer_keys_to_fns = feature_layer_keys_to_fns
        self.max_sequence_size = max_sequence_size
        self.file_io = file_io
        self.all_features = self.feature_config.get_all_features(include_label=False)

        self.feature_layer_map = FeatureLayerMap()
        self.feature_layer_map.add_fns(feature_layer_keys_to_fns)


class UnivariateInteractionModel(InteractionModel):
    """Keras layer that applies in-graph transformations to input feature tensors"""

    def __init__(self,
                 feature_config: FeatureConfig,
                 tfrecord_type: str,
                 feature_layer_keys_to_fns: dict = {},
                 max_sequence_size: int = 0,
                 file_io: FileIO = None,
                 **kwargs):
        """
        Constructor for instantiating a UnivariateInteractionModel

        Parameters
        ----------
        feature_config : `FeatureConfig` object
            FeatureConfig object that defines list of model features
            and the feature transformation functions to be used on each
        tfrecord_type : {"example", "sequence_example"}
            Type of TFRecord protobuf being used for model training
        feature_layer_keys_to_fns : dict
            Dictionary of custom feature transformation functions to be applied
            on the input features
        max_sequence_size : int, optional
            Maximum size of the sequence in SequenceExample protobuf
        file_io : FileIO object
            `FileIO` object that handles read write operations
        """
        super().__init__(feature_config=feature_config,
                         tfrecord_type=tfrecord_type,
                         feature_layer_keys_to_fns=feature_layer_keys_to_fns,
                         max_sequence_size=max_sequence_size,
                         file_io=file_io,
                         **kwargs)

        # Define a one-to-one feature transform mapping
        self.feature_transform_ops = dict()
        for feature_info in self.all_features:
            feature_node_name = feature_info.get(NODE_NAME, feature_info[NAME])
            feature_layer_info = feature_info.get(FEATURE_LAYER_INFO, {})
            if FN in feature_layer_info:
                feature_transform_cls = self.feature_layer_map.get_fn(
                    feature_layer_info[FN])
                if feature_transform_cls:
                    self.feature_transform_ops[feature_node_name] = feature_transform_cls(
                        feature_info=feature_info,
                        file_io=file_io,
                        **kwargs)
                else:
                    raise RuntimeError(
                        "Unsupported feature function: {}".format(feature_layer_info[FN])
                    )

    def call(self, inputs, training=None):
        """
        Apply the feature transform op to each feature

        Parameters
        ----------
        inputs: dict of tensors
            List of tensors that can be found in the FeatureConfig
            key-d with their node_name
        training: boolean
            Boolean specifying if the layer is used in training mode or not

        Returns
        -------
        dict of dict of tensors
            train: dict of tensors
                List of transformed features that are used for training
            metadata: dict of tensors
                List of transformed features that are used as metadata
        """
        train_features = dict()
        metadata_features = dict()

        # Define a dynamic tensor tiling shape
        # NOTE: Can not be hardcoded as we allow for varying sequence_size at inference time
        #
        # TODO: This could possibly be cleaned up now that we have moved to
        #       keras model subclassing API, which allows dynamic graph execution
        #       and doesn't need all this _ahem_ magic
        if self.tfrecord_type == TFRecordTypeKey.SEQUENCE_EXAMPLE:
            train_tile_shape = tf.shape(
                tf.expand_dims(
                    tf.expand_dims(tf.gather(inputs[MASK], indices=0), axis=0), axis=-1
                )
            )
            metadata_tile_shape = tf.shape(
                tf.expand_dims(tf.gather(inputs[MASK], indices=0), axis=0)
            )

        for feature_info in self.all_features:
            feature_node_name = feature_info.get(NODE_NAME, feature_info[NAME])
            feature_layer_info = feature_info[FEATURE_LAYER_INFO]
            feature_tensor = inputs[feature_node_name]

            if feature_node_name in self.feature_transform_ops:
                feature_tensor = self.feature_transform_ops[feature_node_name](
                    feature_tensor, training=training
                )
            elif feature_info[TRAINABLE]:
                # Default feature layer
                feature_tensor = tf.expand_dims(
                    feature_tensor, axis=-1, name="{}_expanded".format(feature_node_name))

            """
            NOTE: If the trainable feature is of type context, then we tile/duplicate
            the values for all examples of the sequence
            """
            if (
                self.tfrecord_type == TFRecordTypeKey.SEQUENCE_EXAMPLE
                and feature_info[TFRECORD_TYPE] == SequenceExampleTypeKey.CONTEXT
            ):
                if feature_info[TRAINABLE]:
                    feature_tensor = tf.tile(feature_tensor, train_tile_shape)
                else:
                    feature_tensor = tf.tile(feature_tensor, metadata_tile_shape)

            if feature_info[TRAINABLE]:
                # Note: All non-string types are converted to float to avoid dtype mismatches.
                # Strings are left as is to be processed by model layers which expect string inputs
                if feature_info[DTYPE] != tf.string:
                    train_features[feature_node_name] = tf.cast(feature_tensor, tf.float32)
                else:
                    train_features[feature_node_name] = feature_tensor
            else:
                if feature_info[DTYPE] == tf.int64:
                    feature_tensor = tf.cast(feature_tensor, tf.float32)
                metadata_features[feature_node_name] = feature_tensor

        return {
            FeatureTypeKey.TRAIN: train_features,
            FeatureTypeKey.METADATA: metadata_features
        }