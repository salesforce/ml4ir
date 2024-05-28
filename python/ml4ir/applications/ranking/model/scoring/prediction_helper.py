import tensorflow as tf


def convert_score_to_rank(features, label, scores):
    """
    Convert scores for each record to rank within query

    Parameters
    ----------
    features: dict of tensors
        Dictionary of tensors that are the features used by the model
    label: Tensor
        Tensor object for the true label
    score: Tensor
        Tensor object containing the scores for each record within a query
        Shape -> [batch_size, sequence_size, 1]

    Returns
    -------
    Tensor
        Rank of each record within a query for all queries in the batch
        Shape -> [batch_size, sequence_size, 1]
    """
    return tf.add(
        tf.argsort(tf.argsort(scores, axis=-1, direction="DESCENDING", stable=True), stable=True),
        tf.constant(1),
    )
