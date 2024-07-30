import tensorflow as tf
from tensorflow.keras import layers
import itertools
from ml4ir.base.config.keys import MonteCarloInferenceKey


class RecordFeatureMask(layers.Layer):
    """
    Mask the record's features of a query batch at the given rate

    Example
    -------
    x = np.ones((2, 5, 4))
        array([[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]],

               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]])
    record_feature_mask = RecordFeatureMask(0.5)
    record_feature_mask(x)
        array([[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [0., 0., 0., 0.],
                [1., 1., 1., 1.]],

               [[0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [0., 0., 0., 0.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]])
    """

    def __init__(self,
                 name="record_feature_mask",
                 mask_rate: float = 0.2,
                 mask_at_inference: bool = False,
                 requires_mask: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        mask_rate: float
            Rate at which the records in the query batch need to be masked to 0s
        mask_at_inference: boolean
            Whether to also mask at inference
            Useful for testing performance at inference time, but should be set to False when training a model to deploy
        requires_mask: bool
            Indicates if the layer requires a mask to be passed to it during forward pass
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        super().__init__(name=name, **kwargs)

        self.mask_rate = mask_rate
        self.mask_at_inference = mask_at_inference
        self.requires_mask = requires_mask

        # Define the probability of picking labels 0 and 1 using the mask_rate
        self.log_odds = tf.math.log([[self.mask_rate, (1. - self.mask_rate)]])

    def call(self, inputs, mask=None, training=None):
        """
        Invoke the record feature mask layer for the input feature tensor

        Parameters
        ----------
        inputs: Tensor object
            Input ranking feature tensor
            Shape: [batch_size, sequence_len, num_features]
        mask: Tensor object
            Mask to be used to identify records to ignore in query (unused in this layer)
            Shape: [batch_size, sequence_len]
        training: bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            Feature tensor where mask_rate of records' features have been masked out (or set to 0.)
            Shape: [batch_size, sequence_len, num_features]
        """
        if self.mask_at_inference or training:
            batch_size = tf.shape(inputs)[0]
            record_dim = tf.shape(inputs)[1]

            batch_log_odds = tf.tile(self.log_odds, [batch_size, 1])
            mask = tf.random.categorical(logits=batch_log_odds,
                                         num_samples=record_dim)
            mask = tf.cast(mask, inputs.dtype)[:, :, tf.newaxis]

            return tf.math.multiply(inputs, mask)
        else:
            return inputs


class QueryFeatureMask(layers.Layer):
    """
    Mask the features of all records per query at the given rate

    Example
    -------
    x = np.ones((2, 5, 4))
        array([[[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]],

               [[1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.],
                [1., 1., 1., 1.]]])
    query_feature_mask = QueryFeatureMask(0.5)
    query_feature_mask(x)
        array([[[1., 1., 0., 1.],
                [1., 1., 0., 1.],
                [1., 1., 0., 1.],
                [1., 1., 0., 1.],
                [1., 1., 0., 1.]],

               [[0., 1., 1., 1.],
                [0., 1., 1., 1.],
                [0., 1., 1., 1.],
                [0., 1., 1., 1.],
                [0., 1., 1., 1.]]])
    """

    # initializing monte carlo config parameters. They are read from model config.
    masking_config = {MonteCarloInferenceKey.FIXED_MASK_COUNT: 1,
                    MonteCarloInferenceKey.USE_FIXED_MASK_IN_TRAINING: False,
                    MonteCarloInferenceKey.USE_FIXED_MASK_IN_TESTING: False}

    def __init__(self,
                 name="query_feature_mask",
                 mask_rate: float = 0.2,
                 mask_at_inference: bool = False,
                 requires_mask: bool = False,
                 **kwargs):
        """
        Parameters
        ----------
        name: str
            Layer name
        mask_rate: float
            Rate at which the records in the query batch need to be masked to 0s
        mask_at_inference: boolean
            Whether to also mask at inference
            Useful for testing performance at inference time, but should be set to False when training a model to deploy
        requires_mask: bool
            Indicates if the layer requires a mask to be passed to it during forward pass
        kwargs:
            Additional key-value args that will be used for configuring the layer
        """
        super().__init__(name=name, **kwargs)

        self.mask_rate = mask_rate
        self.mask_at_inference = mask_at_inference
        self.requires_mask = requires_mask
        self.fixed_masks = None
        self.current_mask_index = -1

    def apply_stochastic_mask(self, inputs, training):
        """
        Apply a stochastic mask to the input tensor.

        This function applies a stochastic mask to the input tensor during training
        or when `mask_at_inference` is True. The mask is generated by comparing random
        uniform values with the mask rate, resulting in a mask tensor of 0s and 1s.
        The mask is applied by element-wise multiplication with the input tensor.

        Parameters:
        inputs (tf.Tensor): A tensor of shape (batch_size, sequence_len, num_features)
                            representing the input data.
        training (bool): A boolean indicating whether the model is in training mode.

        Returns:
        tf.Tensor: A tensor of the same shape as `inputs` with the stochastic mask applied,
                   or the original `inputs` tensor if not applying the mask.

        Notes:
        - The mask is created only during training or if `mask_at_inference` is True.
        - The mask is generated by comparing random values to `self.mask_rate`.

        Example:
        >>> inputs = tf.random.uniform((2, 3, 4))
        >>> masked_inputs = apply_stochastic_mask(inputs, training=True)
        >>> masked_inputs.shape
        TensorShape([2, 3, 4])
        """
        batch_size, sequence_len, num_features = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        if self.mask_at_inference or training:
            mask = tf.cast(tf.math.greater(tf.random.uniform([batch_size, num_features]), self.mask_rate),
                           dtype=tf.float32)
            mask = tf.expand_dims(mask, axis=1)
            mask = tf.tile(mask, [1, sequence_len, 1])
            return tf.multiply(inputs, mask)
        else:
            return inputs

    def apply_fixed_mask(self, inputs, training):
        """
        Apply a fixed mask to the input tensor.

        This function applies a fixed mask to the input tensor based on pre-generated
        masks. If the masks are not yet created, it initializes them. The mask is applied
        by element-wise multiplication with the input tensor.

        Parameters:
        inputs (tf.Tensor): A tensor of shape (batch_size, sequence_len, num_features)
                            representing the input data.
        training (bool): A boolean indicating whether the model is in training mode.

        Returns:
        tf.Tensor: A tensor of the same shape as `inputs` with the fixed mask applied.

        Notes:
        - If the fixed masks are not initialized, they are created using `create_fixed_masks`.
        - The function updates `current_mask_index` to cycle through the masks for each call.

        Example:
        >>> inputs = tf.random.uniform((2, 3, 4))
        >>> masked_inputs = apply_fixed_mask(inputs, training=True)
        >>> masked_inputs.shape
        TensorShape([2, 3, 4])
        """
        batch_size, sequence_len, num_features = tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[2]
        if self.fixed_masks is None:
            self.fixed_masks = self.create_fixed_masks(num_features)
            QueryFeatureMask.masking_config[MonteCarloInferenceKey.FIXED_MASK_COUNT] = len(self.fixed_masks)

        current_mask = tf.expand_dims(tf.expand_dims(self.fixed_masks[self.current_mask_index], 0), 0)
        current_mask = tf.tile(current_mask, [batch_size, sequence_len, 1])
        self.current_mask_index += 1
        self.current_mask_index %= QueryFeatureMask.masking_config[MonteCarloInferenceKey.FIXED_MASK_COUNT]
        return tf.multiply(inputs, tf.cast(current_mask, tf.float32))

    def create_fixed_masks(self, num_features):
        """
        Generate all possible fixed masks for a given feature dimension.

        This function creates a list of tuples representing all possible combinations
        of 0s and 1s for a specified feature dimension. Each tuple corresponds to a
        fixed mask.

        Parameters:
        num_features (int): The dimensionality of the feature space, representing
                           the length of each tuple.

        Returns:
        list of tuples: A list containing all possible fixed masks, where each mask
                        is a tuple of length `num_features` with elements 0 or 1.

        Example:
        >>> create_fixed_masks(2)
        [(0, 0), (0, 1), (1, 0), (1, 1)]
        """
        return list(itertools.product([0, 1], repeat=num_features))

    def call(self, inputs, mask=None, training=None):
        """
        Invoke the query feature mask layer for the input feature tensor

        Parameters
        ----------
        inputs: Tensor object
            Input ranking feature tensor
            Shape: [batch_size, sequence_len, num_features]
        mask: Tensor object
            Mask to be used to identify records to ignore in query (unused in this layer)
            Shape: [batch_size, sequence_len]
        training: bool
            If the layer should be run as training or not

        Returns
        -------
        Tensor object
            Feature tensor where mask_rate of records' features have been masked out (or set to 0.)
            Shape: [batch_size, sequence_len, num_features]
        """
        if training:
            if QueryFeatureMask.masking_config[MonteCarloInferenceKey.USE_FIXED_MASK_IN_TRAINING]:
                return self.apply_fixed_mask(inputs, training)
            else:
                return self.apply_stochastic_mask(inputs, training)
        else:
            if QueryFeatureMask.masking_config[MonteCarloInferenceKey.USE_FIXED_MASK_IN_TESTING]:
                return self.apply_fixed_mask(inputs, training)
            else:
                return self.apply_stochastic_mask(inputs, training)
