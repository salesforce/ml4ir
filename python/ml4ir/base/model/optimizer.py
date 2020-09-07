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
    # Define an exponential learning rate decay schedule
    if learning_rate_decay <= 1.0:
        learning_rate_schedule = ExponentialDecay(
            learning_rate,
            decay_steps=learning_rate_decay_steps,
            decay_rate=learning_rate_decay,
            staircase=True,
        )
    else:
        learning_rate_schedule = learning_rate

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
