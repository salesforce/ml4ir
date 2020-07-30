import tensorflow as tf
from tensorflow import TensorSpec, TensorArray

from ml4ir.base.config.keys import ServingSignatureKey
from ml4ir.base.data.tfrecord_reader import get_parse_fn
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO


def define_default_signature(model, feature_config):
    # Default signature
    #
    # TODO: Define input_signature
    #
    # @tf.function(input_signature=[])
    # def _serve_default(**features):
    #     features_dict = {k: tf.cast(v, tf.float32) for k, v in features.items()}
    #     # Run the model to get predictions
    #     predictions = model(inputs=features_dict)

    #     # Mask the padded records
    #     for key, value in predictions.items():
    #         predictions[key] = tf.where(
    #             tf.equal(features_dict['mask'], 0),
    #             tf.constant(-np.inf),
    #             predictions[key])

    #     return predictions
    return


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
    Add signatures to the tf keras savedmodel

    Returns:
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
    """Defines all serving signatures for the SavedModel"""
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
