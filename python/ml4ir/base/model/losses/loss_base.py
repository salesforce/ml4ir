from tensorflow import keras


class RelevanceLossBase(keras.losses.Loss):
    """
    Defines the loss and last layer activation function used and required by the
    ml4ir.base.model.relevance_model.RelevanceModel.
    """

    def get_final_activation_op(self):
        """
        Returns the final activation layer
        """
        raise NotImplementedError
