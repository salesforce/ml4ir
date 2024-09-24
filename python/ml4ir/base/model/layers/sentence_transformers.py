import json
from pathlib import Path

import tensorflow as tf
import torch
from sentence_transformers.models import Dense as SentenceTransformersDense
from tensorflow.keras.layers import Layer, Dense
from transformers import TFAutoModel, TFBertTokenizer
from sentence_transformers import SentenceTransformer

# NOTE: We set device CPU for the torch backend so that the sentence-transformers model does not use GPU resources
torch.device("cpu")

class SentenceTransformerWithTokenizerLayer(Layer):
    """
    Converts a string tensor into embeddings using sentence transformers
    by first tokenizing the string tensor and then passing through the transformer model

    Some of this code is inspired from -> https://www.philschmid.de/tensorflow-sentence-transformers
    """

    def __init__(self,
                 name="sentence_transformer",
                 model_name_or_path: str = "intfloat/e5-base",
                 trainable: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        model_name_or_path: str
            Name or path to the sentence transformer embedding model
        trainable: bool
            Finetune the pretrained embedding model
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        super().__init__(name=name, **kwargs)

        self.model_name_or_path = Path(model_name_or_path)
        self.st_model = SentenceTransformer(model_name_or_path)
        self.st_model.training = trainable

        self.trainable = trainable

    @classmethod
    def mean_pooling(cls, token_embeddings, attention_mask):
        """Mean pool the token embeddings with the attention mask to generate the embeddings"""
        input_mask_expanded = tf.cast(
            tf.broadcast_to(tf.expand_dims(attention_mask, -1), tf.shape(token_embeddings)),
            tf.float32
        )

        embeddings_sum = tf.math.reduce_sum(token_embeddings * input_mask_expanded, axis=1)
        embeddings_count = tf.clip_by_value(tf.math.reduce_sum(input_mask_expanded, axis=1), 1e-9, tf.float32.max)

        return tf.math.divide_no_nan(embeddings_sum, embeddings_count)

    @classmethod
    def normalize(cls, embeddings):
        """Normalize sentence embeddings"""
        embeddings, _ = tf.linalg.normalize(embeddings, 2, axis=1)
        return embeddings

    def encode(self, inputs, training=None):
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
        # Apply the modules as configured
        return self.st_model.encode(inputs, training=self.trainable)
