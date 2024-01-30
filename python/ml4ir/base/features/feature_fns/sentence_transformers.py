import tensorflow as tf

from ml4ir.base.features.feature_fns.base import BaseFeatureLayerOp
from ml4ir.base.io.file_io import FileIO
from ml4ir.base.model.layers.sentence_transformers import SentenceTransformerWithTokenizerLayer


class SentenceTransformerWithTokenizer(BaseFeatureLayerOp):
    """
    Converts a string tensor into embeddings using sentence transformers
    by first tokenizing the string tensor and then passing through the transformer model

    This is a wrapper around the keras model layer so that it can be used in the feature transform layer
    """
    LAYER_NAME = "sentence_transformer_with_tokenizer"

    MODEL_NAME_OR_PATH = "model_name_or_path"
    TRAINABLE = "trainable"

    def __init__(self, feature_info: dict, file_io: FileIO, **kwargs):
        """
        Initialize layer to define a query length feature transform

        Parameters
        ----------
        feature_info : dict
            Dictionary representing the configuration parameters for the specific feature from the FeatureConfig
        file_io : FileIO object
            FileIO handler object for reading and writing

        Notes
        -----
        Args under feature_layer_info:
            model_name_or_path: str
                Name or path to the sentence transformer embedding model
            finetune_model: bool
                Finetune the pretrained embedding model
        """
        super().__init__(feature_info=feature_info, file_io=file_io, **kwargs)

        self.sentence_transformer_with_tokenizer_op = SentenceTransformerWithTokenizerLayer(
            model_name_or_path=self.feature_layer_args.get(self.MODEL_NAME_OR_PATH, "intfloat/e5-base"),
            trainable=self.feature_layer_args.get(self.TRAINABLE, False)
        )

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
        # Squeeze query dimension as BERTTokenizer only functions on rank 1 tensors
        inputs = tf.squeeze(inputs, axis=-1)

        embedding = self.sentence_transformer_with_tokenizer_op(inputs, training=training)

        # Unsqueeze query dimension
        return tf.expand_dims(embedding, axis=-2)