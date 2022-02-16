import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO
from ml4ir.base.model.architectures.dnn import DNN
from ml4ir.base.model.architectures.transformer import TransformerEncoder


class SetRankLayerKey:
    TRANSFORMER_ENCODER = "transformer_encoder"
    NUM_LAYERS = "num_layers"  # Number of stacked transformer encoder layers
    MODEL_DIM = "model_dim"  # Dimension to use for the transformer inputs and outputs
    NUM_HEADS = "num_heads"  # Number of heads in the multi-head attention layer
    # Number of hidden units in the feed forward layer within the transformer
    FEED_FORWARD_DIM = "feed_forward_dim"
    DROPOUT_RATE = "dropout_rate"


class SetRank(DNN):
    """
    SetRank architecture layer that maps features -> logits
    using a permutation invariant multi-head self attention technique
    inspired from the transformer architecture as described in the following paper

    * Liang Pang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, Jirong Wen. 2020. SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval. In Proceedings of SIGIR '20

    Reference -> https://arxiv.org/pdf/1912.05891.pdf
    """

    def __init__(self,
                 model_config: dict,
                 feature_config: FeatureConfig,
                 file_io: FileIO,
                 **kwargs):
        """
        Initialize a dense neural network layer

        Parameters
        ----------
        model_config: dict
            Dictionary defining the dense architecture spec
        feature_config: FeatureConfig
            FeatureConfig defining how each input feature is used in the model
        file_io: FileIO
            File input output handler
        """
        super().__init__(model_config=model_config,
                         feature_config=feature_config,
                         file_io=file_io,
                         **kwargs)

        self.transformer_encoder_args = model_config[SetRankLayerKey.TRANSFORMER_ENCODER]
        self.num_layers = self.transformer_encoder_args[SetRankLayerKey.NUM_LAYERS]
        self.model_dim = self.transformer_encoder_args[SetRankLayerKey.MODEL_DIM]
        self.num_heads = self.transformer_encoder_args[SetRankLayerKey.NUM_HEADS]
        self.feed_forward_dim = self.transformer_encoder_args[SetRankLayerKey.FEED_FORWARD_DIM]
        self.dropout_rate = self.transformer_encoder_args[SetRankLayerKey.DROPOUT_RATE]

        # Define layers
        self.input_projection_op = layers.Dense(units=self.model_dim)
        # NOTE: We are not using oridnal embeddings for the rank here as described in the original paper
        #       But the current architecture can be easily extended to support it
        self.transformer_encoders = [TransformerEncoder(model_dim=self.model_dim,
                                                        num_heads=self.num_heads,
                                                        feed_forward_dim=self.feed_forward_dim,
                                                        dropout_rate=self.dropout_rate)
                                     for _ in range(self.num_layers)]
        self.dropout = tf.keras.layers.Dropout(self.dropout_rate)

    def get_config(self):
        """Get config for the model"""
        config = super().get_config()
        config[DNNLayerKey.TRANSFORMER_ENCODER] = model_config[DNNLayerKey.TRANSFORMER_ENCODER]
        return config

    def layer_input_op(self, inputs, training=None):
        """
        Generate the input tensor to the DNN layer

        Parameters
        ----------
        inputs: dict of dict of tensors
            Input feature tensors divided as train and metadata
        training: bool
            Boolean to indicate if the layer is used in training or inference mode

        Returns
        -------
        tf.Tensor
            Dense tensor that can be input to the layers of the DNN
        """
        layer_input = super().layer_input_op(inputs, training)

        # Define mask and mask the inputs
        # NOTE: Add extra dimensions to add the padding to the attention logits.
        mask = inputs[FeatureTypeKey.METADATA][FeatureTypeKey.MASK]
        mask = mask[:, tf.newaxis, tf.newaxis, :]

        # Project the features dimension for each result in a query to a fixed model_dim
        x = self.input_projection_op(layer_input, training=training)

        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.transformer_encoders[i](x, training=training, mask=mask)

        return x  # (batch_size, num_results, model_dim)
