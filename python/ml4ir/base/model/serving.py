import tensorflow as tf
from tensorflow import TensorSpec, TensorArray

from ml4ir.base.config.keys import ServingSignatureKey
from ml4ir.base.data.tfrecord_reader import get_parse_fn
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO


def define_default_signature(model, feature_config):
    """Default serving signature to take each model feature as input and outputs the scores"""
    raise NotImplementedError


def define_tfrecord_signature(
    model,
    tfrecord_type: str,
    feature_config: FeatureConfig,
    preprocessing_keys_to_fns: dict,
    postprocessing_fn=None,
    required_fields_only: bool = True,
    pad_sequence: bool = False,
    max_sequence_size: int = 0,
):
    """
    Serving signature that wraps around the keras model trained as a RelevanceModel
    with a pre-step to parse TFRecords and apply additional feature preprocessing

    Parameters
    ----------
    model : keras Model
        Keras model object to be saved
    tfrecord_type : {"example", "sequence_example"}
        Type of the TFRecord protobuf that the saved model will be used on at serving time
    feature_config : `FeatureConfig` object
        FeatureConfig object that defines the input features into the model
        and the corresponding feature preprocesing functions to be used
        in the serving signature
    preprocessing_keys_to_fns : dict
        Dictionary mapping function names to tf.functions that should be saved in the preprocessing step of the tfrecord serving signature
    postprocessing_fn: function
        custom tensorflow compatible postprocessing function to be used at serving time.
        Saved as part of the postprocessing layer of the tfrecord serving signature
    required_fields_only: bool
        boolean value defining if only required fields
        need to be added to the tfrecord parsing function at serving time
    pad_sequence: bool, optional
        Value defining if sequences should be padded for SequenceExample proto inputs at serving time.
        Set this to False if you want to not handle padded scores.
    max_sequence_size : int, optional
        Maximum sequence size for SequenceExample protobuf
        The protobuf object will be padded or clipped to this value

    Returns
    -------
    `tf.function`
        Serving signature function that accepts a TFRecord string tensor and returns predictions
    """

    # TFRecord Signature
    # Define a parsing function for tfrecord protos
    inputs = feature_config.get_all_features(key="node_name", include_label=False)

    """
    NOTE:
    Setting pad_sequence=False for tfrecord signature as it is used at inference time
    and we do NOT want to score on padded records for performance reasons

    Limitation: This limits the serving signature to only run inference on a single query
    at a time given the current implementation. This is a tricky issue to fix because
    there is no real way to generate a dense tensor of ranking scores from different queries,
    as they might have varying number of records in each of them.

    Workaround: To infer on multiple queries, run predict() on each of the queries separately.
    """

    tfrecord_parse_fn = get_parse_fn(
        feature_config=feature_config,
        tfrecord_type=tfrecord_type,
        preprocessing_keys_to_fns=preprocessing_keys_to_fns,
        max_sequence_size=max_sequence_size,
        required_fields_only=required_fields_only,
        pad_sequence=pad_sequence,
    )

    dtype_map = dict()
    for feature_info in feature_config.get_all_features(include_label=False):
        feature_node_name = feature_info.get("node_name", feature_info["name"])
        dtype_map[feature_node_name] = feature_config.get_dtype(feature_info)

    # Define a serving signature for tfrecord
    @tf.function(input_signature=[TensorSpec(shape=[None], dtype=tf.string)])
    def _serve_tfrecord(protos):
        input_size = tf.shape(protos)[0]
        features_dict = {
            feature: TensorArray(dtype=dtype_map[feature], size=input_size) for feature in inputs
        }

        # Define loop index
        i = tf.constant(0)

        # Define loop condition
        def loop_condition(i, protos, features_dict):
            return tf.less(i, input_size)

        # Define loop body
        def loop_body(i, protos, features_dict):
            features, labels = tfrecord_parse_fn(protos[i])
            for feature, feature_val in features.items():
                features_dict[feature] = features_dict[feature].write(i, feature_val)

            i += 1

            return i, protos, features_dict

        # Parse all SequenceExample protos to get features
        _, _, features_dict = tf.while_loop(
            cond=loop_condition, body=loop_body, loop_vars=[i, protos, features_dict],
        )

        # Convert TensorArray to tensor
        features_dict = {k: v.stack() for k, v in features_dict.items()}

        # Run the model to get predictions
        predictions = model(inputs=features_dict)

        # Define a post hook
        if postprocessing_fn:
            predictions = postprocessing_fn(predictions, features_dict)

        return predictions

    return _serve_tfrecord


def define_serving_signatures(
    model,
    tfrecord_type: str,
    feature_config: FeatureConfig,
    preprocessing_keys_to_fns: dict,
    postprocessing_fn=None,
    required_fields_only: bool = True,
    pad_sequence: bool = False,
    max_sequence_size: int = 0,
):
    """
    Defines all serving signatures for the SavedModel

    Parameters
    ----------
    model : keras Model
        Keras model object to be saved
    tfrecord_type : {"example", "sequence_example"}
        Type of the TFRecord protobuf that the saved model will be used on at serving time
    feature_config : `FeatureConfig` object
        FeatureConfig object that defines the input features into the model
        and the corresponding feature preprocesing functions to be used
        in the serving signature
    preprocessing_keys_to_fns : dict
        Dictionary mapping function names to tf.functions that should be saved in the preprocessing step of the tfrecord serving signature
    postprocessing_fn: function
        custom tensorflow compatible postprocessing function to be used at serving time.
        Saved as part of the postprocessing layer of the tfrecord serving signature
    required_fields_only: bool
        boolean value defining if only required fields
        need to be added to the tfrecord parsing function at serving time
    pad_sequence: bool, optional
        Value defining if sequences should be padded for SequenceExample proto inputs at serving time.
        Set this to False if you want to not handle padded scores.
    max_sequence_size : int, optional
        Maximum sequence size for SequenceExample protobuf
        The protobuf object will be padded or clipped to this value

    Returns
    -------
    list of `tf.function`
        List of serving signature functions that make predictions at serving time
        with optional pre and post wrapper steps

    Notes
    -----
    Currently only supports TFRecord serving signature
    """
    return {
        ServingSignatureKey.TFRECORD: define_tfrecord_signature(
            model=model,
            tfrecord_type=tfrecord_type,
            feature_config=feature_config,
            preprocessing_keys_to_fns=preprocessing_keys_to_fns,
            postprocessing_fn=postprocessing_fn,
            required_fields_only=required_fields_only,
            pad_sequence=pad_sequence,
            max_sequence_size=max_sequence_size,
        )
    }
