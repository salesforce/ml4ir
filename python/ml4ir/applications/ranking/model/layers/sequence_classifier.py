import tensorflow as tf
from tensorflow import keras
from transformers import TFAutoModelForSequenceClassification


class SequenceClassifier(keras.layers.Layer):
    """
    A custom layer that wraps a pre-trained sequence classification model from HuggingFace.
    This layer takes input_ids and attention_mask as input and returns classification logits.
    """

    def __init__(
        self,
        model_name_or_path: str,
        trainable: bool = False,
        **kwargs
    ):
        """
        Initialize the SequenceClassifier layer

        Parameters
        ----------
        model_name_or_path : str
            Name or path of the pre-trained model to load
        trainable : bool, optional
            Whether the model should be trainable, by default False
        """
        super().__init__(**kwargs)
        self.model_name_or_path = model_name_or_path
        self.trainable = trainable
        
        # Load the pre-trained model
        self.model = TFAutoModelForSequenceClassification.from_pretrained(
            self.model_name_or_path
        )
        self.model.trainable = self.trainable

    def call(self, inputs, training=False):
        """
        Forward pass through the model

        Parameters
        ----------
        inputs : dict
            Dictionary containing:
            - input_ids: Token IDs tensor
            - attention_mask: Attention mask tensor
            - token_type_ids: Optional token type IDs tensor

        Returns
        -------
        tensor
            Classification logits
        """
        # Ensure token_type_ids is provided (for XLM-R compatibility)
        if 'token_type_ids' not in inputs:
            inputs['token_type_ids'] = tf.zeros_like(inputs['input_ids'])
        
        # Get model outputs
        outputs = self.model(inputs, training=training)
        
        # Return logits
        return tf.reshape(outputs.logits, [-1, 1])

    def get_config(self):
        """Get layer configuration"""
        config = super().get_config()
        config.update({
            "model_name_or_path": self.model_name_or_path,
            "trainable": self.trainable
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Create layer from config"""
        return cls(**config) 