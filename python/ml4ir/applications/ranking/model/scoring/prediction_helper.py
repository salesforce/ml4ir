import tensorflow as tf


def convert_score_to_rank(features, label, scores):

    return tf.add(
        tf.argsort(tf.argsort(scores, axis=-1, direction="DESCENDING", stable=True), stable=True),
        tf.constant(1),
    )
