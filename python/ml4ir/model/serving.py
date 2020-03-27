import tensorflow as tf
from tensorflow import TensorSpec, TensorArray
from ml4ir.config.keys import ServingSignatureKey
from ml4ir.data.tfrecord_reader import make_parse_fn


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


def define_tfrecord_signature(model, feature_config):
    """
    Add signatures to the tf keras savedmodel

    Returns:
        Serving signature function that accepts a TFRecord string tensor and returns predictions
    """

    # TFRecord Signature
    # Define a parsing function for tfrecord protos
    inputs = feature_config.get_all_features(key="node_name", include_label=False) + [
        "num_records"
    ]
    tfrecord_parse_fn = make_parse_fn(
        feature_config=feature_config, max_num_records=25, required_only=True
    )

    # Define a serving signature for tfrecord
    @tf.function(input_signature=[TensorSpec(shape=[None], dtype=tf.string)])
    def _serve_tfrecord(sequence_example_protos):
        input_size = tf.shape(sequence_example_protos)[0]
        features_dict = {
            feature: TensorArray(dtype=tf.float32, size=input_size) for feature in inputs
        }

        # Define loop index
        i = tf.constant(0)

        # Define loop condition
        def loop_condition(i, sequence_example_protos, features_dict):
            return tf.less(i, input_size)

        # Define loop body
        def loop_body(i, sequence_example_protos, features_dict):
            features, labels = tfrecord_parse_fn(sequence_example_protos[i])
            for feature, feature_val in features.items():
                features_dict[feature] = features_dict[feature].write(
                    i, tf.cast(feature_val, tf.float32)
                )

            i += 1

            return i, sequence_example_protos, features_dict

        # Parse all SequenceExample protos to get features
        _, _, features_dict = tf.while_loop(
            cond=loop_condition,
            body=loop_body,
            loop_vars=[i, sequence_example_protos, features_dict],
        )

        # Convert TensorArray to tensor
        features_dict = {k: v.stack() for k, v in features_dict.items()}

        # Run the model to get predictions
        predictions = model(inputs=features_dict)

        # Mask the padded records
        for key, value in predictions.items():
            predictions[key] = tf.where(
                tf.equal(features_dict["mask"], 0), tf.constant(0.0), predictions[key]
            )

        return predictions

    return _serve_tfrecord


def define_serving_signatures(model, feature_config):
    """Defines all serving signatures for the SavedModel"""
    return {
        # ServingSignatureKey.DEFAULT: define_default_signature(
        #     model, feature_config),
        ServingSignatureKey.TFRECORD: define_tfrecord_signature(model, feature_config)
    }
