import json
from pathlib import Path

import os
import tensorflow as tf
import torch
from sentence_transformers.models import Dense as SentenceTransformersDense
from tensorflow.keras.layers import Layer, Dense
from transformers import TFAutoModel, TFBertTokenizer
from sentence_transformers import SentenceTransformer

# NOTE: We set device CPU for the torch backend so that the sentence-transformers model does not use GPU resources
torch.device("cpu")


class SentenceTransformerLayerKey:
    """Stores the names of the sentence-transformer model layers"""
    TRANSFORMER = "sentence_transformers.models.Transformer"
    POOLING = "sentence_transformers.models.Pooling"
    DENSE = "sentence_transformers.models.Dense"
    NORMALIZE = "sentence_transformers.models.Normalize"


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
        if not Path(model_name_or_path).exists():  # The user provided a model name
            # If the sentence_transformer model files are not present, we initialize it to trigger a download
            st_model = SentenceTransformer(model_name_or_path)

            print("printing torch home path")
            print(torch.hub._get_torch_home())
            print(list(os.walk(torch.hub._get_torch_home())))
            self.model_name_or_path = Path(
                torch.hub._get_torch_home()) / "sentence_transformers" / model_name_or_path.replace("/", "_")
            if not self.model_name_or_path.exists():
                raise FileNotFoundError(
                    f"{self.model_name_or_path} does not exist. Verify the `model_name_or_path` argument")

            del st_model

        # Load the modules.json config to add custom model layers
        self.modules = json.load(open(self.model_name_or_path / "modules.json"))
        self.module_names = [module["name"] for module in self.modules]

        self.trainable = trainable

        # Define tokenizer as part of the tensorflow graph
        self.tokenizer = TFBertTokenizer.from_pretrained(self.model_name_or_path, **kwargs)

        self.transformer_model = None
        self.dense = None
        self.apply_dense = False
        self.normalize_embeddings = False
        self.pool_embeddings = False
        for module in self.modules:
            # Define the transformer model and initialize pretrained weights
            if module["type"] == SentenceTransformerLayerKey.TRANSFORMER:
                self.transformer_model = TFAutoModel.from_pretrained(self.model_name_or_path,
                                                                     from_pt=True,
                                                                     trainable=self.trainable,
                                                                     **kwargs)

            # Define mean pooling op
            if module["type"] == SentenceTransformerLayerKey.POOLING:
                self.pool_embeddings = True

            # Define normalization op
            if module["type"] == SentenceTransformerLayerKey.NORMALIZE:
                self.normalize_embeddings = True

            # Define dense layer if present in the model and initialize pretrained weights
            if module["type"] == SentenceTransformerLayerKey.DENSE:
                self.apply_dense = True
                st_dense = SentenceTransformersDense.load(self.model_name_or_path / module["path"])
                self.dense = Dense(units=st_dense.get_config_dict()["out_features"],
                                   kernel_initializer=tf.keras.initializers.Constant(
                                       st_dense.state_dict()["linear.weight"].T),
                                   bias_initializer=tf.keras.initializers.Constant(
                                       st_dense.state_dict()["linear.bias"]),
                                   activation=st_dense.get_config_dict()["activation_function"].split(".")[-1].lower(),
                                   trainable=self.trainable)
                del st_dense

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
        # Tokenize string tensors
        tokens = self.tokenizer(inputs)
        # NOTE: TFDistilBertModel does not expect the token_type_ids key
        tokens.pop("token_type_ids", None)

        # Apply the modules as configured
        embeddings = self.transformer_model(tokens, training=training)

        if self.pool_embeddings:
            embeddings = self.mean_pooling(embeddings[0], tokens["attention_mask"])

        if self.apply_dense:
            embeddings = self.dense(embeddings, training=training)

        if self.normalize_embeddings:
            embeddings = self.normalize(embeddings)

        return embeddings
