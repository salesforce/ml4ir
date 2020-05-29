import tensorflow as tf
from tensorflow import keras
from typing import List, Dict
import numpy as np
from tensorflow.keras import backend as K

from ml4ir.base.config.keys import TFRecordTypeKey


def get_predict_fn(
    model: keras.Model,
    tfrecord_type: str,
    feature_config: Dict,
    inference_signature: str = "serving_default",
    is_compiled: bool = False,
    output_name: str = "relevance_score",
    features_to_return: List = [],
    additional_features: Dict = {},
    max_sequence_size: int = 0,
):
    # Load the forward pass function for the tensorflow model
    if is_compiled:
        infer = model
    else:
        # If SavedModel was loaded without compilation
        infer = model.signatures[inference_signature]

    # Get features to log
    features_to_log = [f.get("node_name", f["name"]) for f in features_to_return] + [output_name]

    @tf.function
    def _flatten_records(x):
        """Collapse first two dimensions -> [batch_size, max_sequence_size]"""
        return tf.reshape(x, tf.concat([[-1], tf.shape(x)[2:]], axis=0))

    @tf.function
    def _filter_records(x, mask):
        """Filter records that were padded in each query"""
        return tf.squeeze(
            tf.gather_nd(
                x, tf.where(tf.not_equal(tf.cast(mask, tf.int64), tf.constant(0, dtype="int64"))),
            )
        )

    @tf.function
    def _predict_score(features, label):
        # features = {k: tf.cast(v, tf.float32) for k, v in features.items()}
        if is_compiled:
            scores = infer(features)[output_name]
        else:
            scores = infer(**features)[output_name]

        # Set scores of padded records to 0
        if tfrecord_type == TFRecordTypeKey.SEQUENCE_EXAMPLE:
            scores = tf.where(tf.equal(features["mask"], 0), tf.constant(-np.inf), scores)

            mask = _flatten_records(features["mask"])

        predictions_dict = dict()
        for feature_name in features_to_log:
            if feature_name == feature_config.get_label(key="node_name"):
                feat_ = label
            elif feature_name == output_name:
                feat_ = scores
            else:
                if feature_name in features:
                    feat_ = features[feature_name]
                else:
                    raise KeyError("{} was not found in input training data".format(feature_name))

            predictions_dict[feature_name] = feat_

        # Process additional features
        for feature_name, feature_fn in additional_features.items():
            predictions_dict[feature_name] = feature_fn(features, label, scores)

        # Explode context features to each record for logging
        # NOTE: This assumes that the record dimension is on axis 1, like previously
        for feature_name in predictions_dict.keys():
            feat_ = predictions_dict[feature_name]
            if tfrecord_type == TFRecordTypeKey.SEQUENCE_EXAMPLE:
                feat_ = tf.cond(
                    tf.equal(tf.shape(feat_)[1], tf.constant(1)),
                    true_fn=lambda: K.repeat_elements(feat_, rep=max_sequence_size, axis=1),
                    false_fn=lambda: feat_,
                )

                # Collapse from one query per data point to one record per data point
                # and remove padded dummy records
                feat_ = _filter_records(_flatten_records(feat_), mask)

            predictions_dict[feature_name] = tf.squeeze(feat_)

        return predictions_dict

    return _predict_score
