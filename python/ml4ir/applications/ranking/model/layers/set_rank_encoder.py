import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_models as tfm


class SetRankEncoderLayerKey:
    REQUIRES_MASK = "requires_mask"  # Indicates if the layer requires a mask to be passed to it during forward pass
    ENCODING_SIZE = "encoding_size"  # Size of the projection which will serve as both the input and output size to the encoder
    PROJECTION_DROPOUT = "projection_dropout"  # Dropout rate to be applied after the input projection layer


class SetRankEncoder(layers.Layer):
    """
    SetRank architecture layer that maps features for a document -> encoding
    using a permutation invariant multi-head self attention technique attending to all documents in the query.

    Inspired from the transformer architecture as described in the following paper
    * Liang Pang, Jun Xu, Qingyao Ai, Yanyan Lan, Xueqi Cheng, Jirong Wen. 2020.
    SetRank: Learning a Permutation-Invariant Ranking Model for Information Retrieval. In Proceedings of SIGIR '20

    Reference -> https://arxiv.org/pdf/1912.05891.pdf
    """

    def __init__(self, **kwargs):
        """
        For a full list of args that can be passed to configure this layer, please check the below official layer doc
        https://www.tensorflow.org/api_docs/python/tfm/nlp/models/TransformerEncoder
        """
        super(SetRankEncoder, self).__init__()

        self.requires_mask = kwargs.pop(SetRankEncoderLayerKey.REQUIRES_MASK, False)
        assert self.requires_mask, "To use SetRankEncoder layer, the `requires_mask` arg needs to be set to true"

        self.encoding_size = kwargs.pop(SetRankEncoderLayerKey.ENCODING_SIZE)
        self.projection_dropout_rate = kwargs.pop(SetRankEncoderLayerKey.PROJECTION_DROPOUT, 0.0)

        self.input_projection_op = layers.Dense(units=self.encoding_size)
        self.projection_dropout_op = layers.Dropout(rate=self.projection_dropout_rate)
        self.transformer_encoder = tfm.nlp.models.TransformerEncoder(**kwargs)

    def call(self, inputs, mask, training=None):
        """
        Invoke the set transformer encoder (permutation invariant) for the input feature tensor

        Parameters
        ----------
        inputs : Tensor object
            Input ranking feature tensor
            Shape: [batch_size, sequence_len, num_features]
        mask: Tensor object
            Mask to be used as the attention mask for the TransformerEncoder
            to indicate which documents to not attend to in the query
            Shape: [batch_size, sequence_len]
        training : bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            Set transformer encoder (permutation invariant) output tensor
            Shape: [batch_size, sequence_len, encoding_size]
        """
        # Project input from shape
        # [batch_size, sequence_len, num_features] -> [batch_size, sequence_len, encoding_size]
        encoder_inputs = self.input_projection_op(inputs, training=training)
        encoder_inputs = self.projection_dropout_op(encoder_inputs, training=training)

        # Convert 2D mask to 3D mask to be used for attention
        attention_mask = tf.matmul(mask[:, :, tf.newaxis], mask[:, tf.newaxis, :])

        encoder_output = self.transformer_encoder(encoder_inputs=encoder_inputs,
                                                  attention_mask=attention_mask,
                                                  training=training)

        return encoder_output

    def get_config(self):
        config = self.transformer_encoder.get_config()
        config.update({
            SetRankEncoderLayerKey.ENCODING_SIZE: self.encoding_size,
            SetRankEncoderLayerKey.PROJECTION_DROPOUT: self.projection_dropout_rate
        })

        return config
