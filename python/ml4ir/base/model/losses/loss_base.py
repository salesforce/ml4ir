from tensorflow.keras import layers


class RelevanceLossBase(layers.Layer):
    """
    Abstract class that defines the loss and final activation function
    used to train a RelevanceModel
    """

    def call(self, inputs, y_true, y_pred, training=None):
        """
        Compute the loss using predicted probabilities and expected labels

        Parameters
        ----------
        inputs: dict of dict of tensors
            Dictionary of input feature tensors
        y_true: tensor
            True labels
        y_pred: tensor
            Predicted scores
        training: boolean
            Boolean indicating whether the layer is being used in training mode

        Returns
        -------
        tensor
            Resulting loss tensor after applying comparing the y_pred and y_true values
        """
        raise NotImplementedError

    def final_activation_op(self, inputs, training=None):
        """
        Final activation layer that is applied to the logits tensor to get the scores

        Parameters
        ----------
        inputs: dict of dict of tensors
            Dictionary of input feature tensors with scores
        training: boolean
            Boolean indicating whether the layer is being used in training mode

        Returns
        -------
        tensor
            Resulting score tensor after applying the function on the logits
        """
        raise NotImplementedError

    def get_config(self):
        """Return layer config that is used while serialization"""
        config = super().get_config()
        config.update({
            "output_name": self.output_name
        })
        return config
