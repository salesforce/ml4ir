import tensorflow as tf
from tensorflow.keras import layers
from transformers import TFAutoModel, TFBertTokenizer, AutoTokenizer
from sentence_transformers import SentenceTransformer


class SentenceTransformerWithTokenizerLayer(layers.Layer):
    """
    Converts a string tensor into embeddings using sentence transformers
    by first tokenizing the string tensor and then passing through the transformer model

    Some of this code is inspired from -> https://www.philschmid.de/tensorflow-sentence-transformers
    """

    def __init__(self,
                 name="sentence_transformer",
                 model_name_or_path: str = "intfloat/e5-base",
                 load_model_from_pt: bool = True,
                 normalize_embeddings: bool = False,
                 finetune_model: bool = False,
                 run_sanity_check: bool = True,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        model_name_or_path: str
            Name or path to the sentence transformer embedding model
        load_model_from_pt: bool
            Whether to load the model from pytorch or tensorflow
        normalize_embeddings: bool
            Whether to normalize the final sentence embeddings
            Some sentence transformer models use normalization
        finetune_model: bool
            Finetune the pretrained embedding model
        run_sanity_check: bool
            Flag to indicate whether the model should be sanity checked with the Torch model
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        self.model_name_or_path = model_name_or_path
        self.load_model_from_pt = load_model_from_pt
        self.normalize_embeddings = normalize_embeddings
        self.finetune_model = finetune_model
        self.run_sanity_check = run_sanity_check

        self.tokenizer = TFBertTokenizer.from_pretrained(self.model_name_or_path, **kwargs)
        self.sentence_transformer = TFAutoModel.from_pretrained(self.model_name_or_path,
                                                                from_pt=self.load_model_from_pt,
                                                                trainable=self.finetune_model,
                                                                **kwargs)

        # TODO: Add a sanity check with the PyTorch model

        super().__init__(name=name, **kwargs)

    @classmethod
    def mean_pooling(cls, model_output, attention_mask):
        """Mean pool the hidden state with the attention mask to generate the embeddings"""
        token_embeddings = model_output[0] # First element of model_output contains all token embeddings
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

        # Forward pass through sentence-transformers model
        model_output = self.sentence_transformer(tokens, training=training)

        # Mean pooling to get embeddings
        embeddings = self.mean_pooling(model_output, tokens["attention_mask"])

        # Normalize if configured
        if self.normalize_embeddings:
            embeddings = self.normalize(embeddings)

        return embeddings
