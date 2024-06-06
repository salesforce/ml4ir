import tensorflow as tf

from ml4ir.applications.ranking.model.layers.rank_transform import ReciprocalRankLayer
from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.base.io.file_io import FileIO


class ReciprocalRank(BaseFeatureLayerOp):
    """
    Converts a tensor of scores into reciprocal ranks.
    Can optionally add a constant or variable k to the rank

    Final formulation of reciprocal rank = 1 / (k + rank)
    """
    LAYER_NAME = "reciprocal_rank"

    K = "k"
    K_TRAINABLE = "k_trainable"
    IGNORE_ZERO_SCORE = "ignore_zero_score"
    SCALE_TO_ONE = "scale_to_one"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize layer to define a reciprocal rank layer

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_info["feature_layer_info"]:
            k : float
                Constant value to be added to the rank before reciprocal
            k_trainable: bool
                If k should be a learnable variable; will be initialized with value of k
            ignore_zero_score: bool
                Use zero reciprocal rank for score value of 0.0
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.reciprocal_rank_op = ReciprocalRankLayer(
            k=self.feature_layer_args.get(self.K, 0.),
            k_trainable=self.feature_layer_args.get(self.K_TRAINABLE, False),
            ignore_zero_score=self.feature_layer_args.get(self.IGNORE_ZERO_SCORE, True),
            scale_to_one=self.feature_layer_args.get(self.SCALE_TO_ONE, False))

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
        reciprocal_ranks = self.reciprocal_rank_op(inputs, training)

        return tf.expand_dims(reciprocal_ranks, axis=-1)
