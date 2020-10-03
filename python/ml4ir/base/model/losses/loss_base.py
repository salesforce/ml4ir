class RelevanceLossBase(object):
    """
    Abstract class that defines the loss and final activation function
    used to train a RelevanceModel
    """

    def get_loss_fn(self, **kwargs):
        """
        Returns the loss function _loss_fn()

        Parameters
        ----------
        kwargs : dict
            Additional key value arguments can be passed as needed.
            For example, metadata features can be passed to compute custom losses

        Returns
        -------
        function
            Loss function that computes the loss from predicted scores
            and true labels
        """

        def _loss_fn(y_true, y_pred):
            pass

        return _loss_fn

    def get_final_activation_op(self):
        """
        Returns the final activation layer

        Returns
        -------
        function
            Final activation function that is applied on the scores
            before computing losses
        """
        raise NotImplementedError
