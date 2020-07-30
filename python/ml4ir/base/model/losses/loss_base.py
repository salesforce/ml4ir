class RelevanceLossBase(object):
    """
    Defines the loss and last layer activation function used and required by the
    ml4ir.base.model.relevance_model.RelevanceModel.
    """

    def get_loss_fn(self, **kwargs):
        """
        Returns the loss function _loss_fn()
        """
        def _loss_fn(y_true, y_pred):
            pass

        return _loss_fn

    def get_final_activation_op(self):
        """
        Returns the final activation layer
        """
        raise NotImplementedError
