import tensorflow as tf

from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.applications.ranking.model.layers.normalization import TheoreticalMinMaxNormalization as TheoreticalMinMaxNormalizationLayer
from ml4ir.base.io.file_io import FileIO


class TheoreticalMinMaxNormalization(BaseFeatureLayerOp):
    """
    Min Max Normalization of individual query features,
    where the theoretical min is used instead of the minimum.

    Reference -> An Analysis of Fusion Functions for Hybrid Retrieval
                 https://arxiv.org/abs/2210.11934
    """
    LAYER_NAME = "theoretical_min_max_norm"

    THEORETICAL_MIN = "theoretical_min"


    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize layer to define a theoretical min max normalization

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            theoretical_min : float
                Theoretical minimum to use for the query's record features
                Default value of 0. is used if not specified.
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.theoretical_min = self.feature_layer_args.get(self.THEORETICAL_MIN, 0.)
        self.tmm_norm_op = TheoreticalMinMaxNormalizationLayer(theoretical_min=self.theoretical_min)

    def call(self, inputs, training=None):
        """
        Defines the forward pass for the layer on the inputs tensor

        Parameters
        ----------
        inputs: tensor
            Input tensor on which the feature transforms are applied
        training: boolean
            Boolean flag indicating if the layer is being used in training mode or not

        Returns
        -------
        tf.Tensor
            Resulting tensor after the forward pass through the feature transform layer
        """
        normed_inputs = self.tmm_norm_op(inputs, training)

        return tf.expand_dims(normed_inputs, axis=-1)