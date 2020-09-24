import tensorflow.keras.optimizers as tf_optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay

from ml4ir.base.config.keys import OptimizerKey


def get_optimizer(
    optimizer_key: str = "SGD",
    with_exp_decay: bool = False,
    initial_learning_rate: float = 0.01,
    learning_rate_decay: float = 1.0,
    learning_rate_decay_steps: int = 1000000,
    **kwargs
) -> tf_optimizers.Optimizer:
    """
    This function defines the optimizer used by ml4ir.
    By default, if no arguments are passed the function returns
    vanilla SGD. Given a valid optimizer key, the function will return
    the passed optimizer, with its default params. You may pass any other
    valid arguments to the function to control aspects of the optimizer,
    according to the official Tensorflow documentation. See References below.

    Users may use an ExponentialDecay learning rate schedule.

    Arguments:
        optimizer_key: string optimizer name to be used as defined
         under tensorflow.keras.optimizers
        with_exp_decay: Set to True, to add Exponential Decay schedule
        learning_rate_decay,learning_rate_decay, learning_rate_decay_steps:
        Params controlling the decay schedule,
         cf. https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay

    You may pass any other valid arguments to the function to control aspects of the optimizer,
    according to the official tensforflow documentation.

    References:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay


    Examples:
        get_optimizer() -> SGD optimizer
        get_optimizer('Adam', learning_rate=0.002) -> Adam, with lr=0.002
    FIXME:
        Define all arguments overriding tensorflow defaults in a separate file
        for visibility with ml4ir users

    """
    # Define an exponential learning rate decay schedule
    learning_rate_schedule = ExponentialDecay(
        initial_learning_rate,
        decay_steps=learning_rate_decay_steps,
        decay_rate=learning_rate_decay,
        staircase=True,
    )

    try:
        optimizer = getattr(tf_optimizers, optimizer_key)  # returns the class
    except AttributeError:
        raise AttributeError("ml4ir uses 'tensorflow.keras.optimizers' as a factory of optimizers. "
                             "You provided {}, which does not exist in that module.".format(optimizer_key))

    if with_exp_decay:
        return optimizer(learning_rate = learning_rate_schedule, **kwargs)
    return optimizer(**kwargs)
