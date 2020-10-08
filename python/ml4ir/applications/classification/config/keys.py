from ml4ir.base.config.keys import Key


class LossKey(Key):
    """Model loss keys that can be used with classification model"""

    CATEGORICAL_CROSS_ENTROPY = "categorical_cross_entropy"
    SIGMOID_CROSS_ENTROPY = "sigmoid_cross_entropy"


class MetricKey(Key):
    """Model metric keys that can be used with classification model"""

    CATEGORICAL_ACCURACY = "categorical_accuracy"
    TOP_5_CATEGORICAL_ACCURACY = "top_5_categorical_accuracy"
