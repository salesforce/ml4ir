import tensorflow.keras.optimizers as tf_optimizers
from ml4ir.config.keys import OptimizerKey


def get_optimizer(
    optimizer_key: str, learning_rate: float, learning_rate_decay: float
) -> tf_optimizers.Optimizer:
    if optimizer_key == OptimizerKey.ADAM:
        return tf_optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_key == OptimizerKey.NADAM:
        return tf_optimizers.Nadam(learning_rate=learning_rate)
    elif optimizer_key == OptimizerKey.ADAGRAD:
        return tf_optimizers.Adagrad(learning_rate=learning_rate)
    elif optimizer_key == OptimizerKey.SGD:
        return tf_optimizers.SGD(learning_rate=learning_rate)
    elif optimizer_key == OptimizerKey.RMS_PROP:
        return tf_optimizers.RMSprop(learning_rate=learning_rate)
    else:
        raise ValueError("illegal Optimizer key: " + optimizer_key)
