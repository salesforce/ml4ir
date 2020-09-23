import tensorflow.keras.optimizers as tf_optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from ml4ir.base.config.keys import OptimizerKey


def get_optimizer(
    optimizer_key: str,
    learning_rate: float,
    learning_rate_decay: float = 1.0,
    learning_rate_decay_steps: int = 1000000,
    gradient_clip_value: float = 1000000,
) -> tf_optimizers.Optimizer:
    """
    This function defines the optimizer used by ml4ir.
    Users have the option to define an ExponentialDecay learning rate schedule

    Arguments:
        optimizer_key: string optimizer name to be used as defined under ml4ir.base.config.keys.OptimizerKey
        learning_rate: floating point learning rate for the optimizer
        learning_rate_decay: floating point rate at which the learning rate will be decayed every learning_rate_decay_steps
        learning_rate_decay_steps: int representing number of iterations after which learning rate will be decreased exponentially
        gradient_clip_value: float value representing the clipvalue for gradient updates. Not setting this to a reasonable value based on the model will lead to gradient explosion and NaN losses.

    References:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay

    FIXME:
        Define all arguments overriding tensorflow defaults in a separate file
        for visibility with ml4ir users
    """
    # Define an exponential learning rate decay schedule
    learning_rate_schedule = ExponentialDecay(
        learning_rate,
        decay_steps=learning_rate_decay_steps,
        decay_rate=learning_rate_decay,
        staircase=True,
    )

    if optimizer_key == OptimizerKey.ADAM:
        return tf_optimizers.Adam(
            learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value
        )
    elif optimizer_key == OptimizerKey.NADAM:
        return tf_optimizers.Nadam(
            learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value
        )
    elif optimizer_key == OptimizerKey.ADAGRAD:
        return tf_optimizers.Adagrad(
            learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value
        )
    elif optimizer_key == OptimizerKey.SGD:
        return tf_optimizers.SGD(
            learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value
        )
    elif optimizer_key == OptimizerKey.RMS_PROP:
        return tf_optimizers.RMSprop(
            learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value
        )
    else:
        raise ValueError("illegal Optimizer key: " + optimizer_key)
