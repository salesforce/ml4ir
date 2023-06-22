import tensorflow as tf

from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.base.io.file_io import FileIO


class StringMultiLabelProcessor(BaseFeatureLayerOp):
    """
    Process the label string to generate a weighted sum of the ranking labels that can be used to train and test

    Example
    -------
    x = <>
    string_label_multi_processor = StringMultiLabelProcessor(<>)
    string_label_multi_processor(x)
    <>
    """
    LAYER_NAME = "string_multi_label_processor"
    SEPERATOR = "separator"
    NUM_LABELS = "num_labels"
    BINARIZE = "binarize"
    LABEL_WEIGHTS = "label_weights"
    TEST_LABEL_WEIGHTS = "test_label_weights"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize the feature layer

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the feature_config for the input feature
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            sep: str
                Separator to split on to create multi label
            num_labels: int
                Number of labels
            binarize: bool
                If the individual labels should be binarized to 0 and 1
            label_weights: list
                List of weights to be used to combine the labels
                If unspecifed, equal weights of 1. will be used to combine
            test_label_weights: list
                List of weights to be used to combine the labels at test time
                If unspecifed, label_weights will be used to combine
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.separator = self.feature_layer_args.get(self.SEPERATOR, "_")
        self.num_labels = self.feature_layer_args.get(self.NUM_LABELS, None)
        assert self.num_labels, "The num_labels argument needs to be specified in order to handle missing labels"
        self.default_value = "_".join(["0"] * self.num_labels)
        self.binarize = self.feature_layer_args.get(self.BINARIZE, True)

        self.label_weights = self.feature_layer_args.get(self.LABEL_WEIGHTS, None)
        if not self.label_weights:
            self.label_weights = [1.0] * self.num_labels
        self.test_label_weights = self.feature_layer_args.get(self.TEST_LABEL_WEIGHTS, self.label_weights)
        assert len(self.label_weights) == len(
            self.test_label_weights), "The label_weights and test_label_weights arguments should contain the same number of weights"

        self.label_weights = tf.constant(self.label_weights, tf.float32)
        self.test_label_weights = tf.constant(self.test_label_weights, tf.float32)

    def call(self, inputs, training=None):
        """
        Invoke the string multi label processor

        Parameters
        ----------
        inputs: Tensor object
            Input ranking feature tensor
            Shape: [batch_size, sequence_len, feature_dim]
        training: bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            Feature tensor where the string label is parsed and aggregated
            into a single label based on the weights provided
            Shape: [batch_size, sequence_len, feature_dim]
        """
        # Replace empty strings with 0s
        label = tf.strings.regex_replace(inputs, r"^$", self.default_value)

        # Split and convert to numeric label tensor
        label = tf.strings.split(label, sep=self.separator, maxsplit=self.num_labels)
        label = tf.strings.to_number(label, tf.float32).to_tensor()

        # Binarize the multi labels if greater than 0
        if self.binarize:
            label = tf.cast(label > 0., tf.float32)

        # Weighted sum of the labels with the specified weights
        if training:
            label = tf.reduce_sum(tf.multiply(label, self.label_weights), axis=-1)
        else:
            label = tf.reduce_sum(tf.multiply(label, self.test_label_weights), axis=-1)

        # Adjusting the shape to comply with the label shape
        label = tf.squeeze(label, axis=-1)
        return label
