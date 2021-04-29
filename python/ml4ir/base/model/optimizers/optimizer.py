import tensorflow.keras.optimizers as tf_optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from ml4ir.base.model.optimizers import cyclic_learning_rate
from ml4ir.base.config.keys import OptimizerKey, LearningRateScheduleKey, CyclicLearningRateType
import tensorflow as tf


class OptimizerDefaultValues(object):
    """Default values for unspecified parameters of the optimizer and learning rate schedule."""

    CONSTANT_LR = 0.01
    EXP_DECAY_STEPS = 100000
    EXP_DECAY_RATE = 0.96
    CYCLIC_INITIAL_LEARNING_RATE = 0.001
    CYCLIC_MAXIMAL_LEARNING_RATE = 0.01
    CYCLIC_STEP_SIZE = 10
    CYCLIC_GAMMA = 1.0


def choose_optimizer(model_config, learning_rate_schedule):
    """
    Define the optimizer used for training the RelevanceModel
    Users have the option to define an ExponentialDecay learning rate schedule

    Parameters
    ----------
    model_config : dict
        model configuration doctionary

    Returns
    -------
    tensorflow optimizer

    Notes
    -----
    References:
    - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
    - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
    - https://arxiv.org/pdf/1506.01186.pdf
    """

    if 'optimizer' not in model_config:
        return tf_optimizers.Adam(learning_rate=learning_rate_schedule, clipvalue=5.0)
    else:
        optimizer_key = model_config['optimizer']['key']
        if 'gradient_clip_value' in model_config['optimizer']:
            config = {'learning_rate': learning_rate_schedule,
                      'clipvalue': model_config['optimizer']['gradient_clip_value']}
        else:
            config = {'learning_rate': learning_rate_schedule}
        return tf.keras.optimizers.get({'class_name': optimizer_key, 'config': config})


def choose_scheduler(model_config):
    """
    Define the optimizer used for training the RelevanceModel
    Users have the option to define an ExponentialDecay learning rate schedule

    Parameters
    ----------
    model_config : dict
        model configuration doctionary

    Returns
    -------
    tensorflow learning rate scheduler

    Notes
    -----
    References:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
        https://arxiv.org/pdf/1506.01186.pdf
    """

    if 'lr_schedule' not in model_config:
        # use constant lr schedule
        learning_rate_schedule = ExponentialDecay(
            initial_learning_rate=OptimizerDefaultValues.CONSTANT_LR,
            decay_steps=10000000,
            decay_rate=1.0,
        )

    else:
        lr_schedule = model_config['lr_schedule']
        lr_schedule_key = lr_schedule['key']

        if lr_schedule_key == LearningRateScheduleKey.EXPONENTIAL:
            learning_rate_schedule = ExponentialDecay(
                initial_learning_rate=lr_schedule.get(
                    'learning_rate', OptimizerDefaultValues.CONSTANT_LR),
                decay_steps=lr_schedule.get(
                    'learning_rate_decay_steps', OptimizerDefaultValues.EXP_DECAY_STEPS),
                decay_rate=lr_schedule.get(
                    'learning_rate_decay', OptimizerDefaultValues.EXP_DECAY_RATE),
                staircase=True,
            )

        elif lr_schedule_key == LearningRateScheduleKey.CONSTANT:
            learning_rate_schedule = ExponentialDecay(
                initial_learning_rate=lr_schedule.get(
                    'learning_rate', OptimizerDefaultValues.CONSTANT_LR),
                decay_steps=10000000,
                decay_rate=1.0,
            )

        elif lr_schedule_key == LearningRateScheduleKey.CYCLIC:
            lr_schedule_type = lr_schedule['type']
            if lr_schedule_type == CyclicLearningRateType.TRIANGULAR:
                learning_rate_schedule = cyclic_learning_rate.TriangularCyclicalLearningRate(
                    initial_learning_rate=lr_schedule.get(
                        'initial_learning_rate', OptimizerDefaultValues.CYCLIC_INITIAL_LEARNING_RATE),
                    maximal_learning_rate=lr_schedule.get(
                        'maximal_learning_rate', OptimizerDefaultValues.CYCLIC_MAXIMAL_LEARNING_RATE),
                    step_size=lr_schedule.get(
                        'step_size', OptimizerDefaultValues.CYCLIC_STEP_SIZE),
                )
            elif lr_schedule_type == CyclicLearningRateType.TRIANGULAR2:
                learning_rate_schedule = cyclic_learning_rate.Triangular2CyclicalLearningRate(
                    initial_learning_rate=lr_schedule.get(
                        'initial_learning_rate', OptimizerDefaultValues.CYCLIC_INITIAL_LEARNING_RATE),
                    maximal_learning_rate=lr_schedule.get(
                        'maximal_learning_rate', OptimizerDefaultValues.CYCLIC_MAXIMAL_LEARNING_RATE),
                    step_size=lr_schedule.get(
                        'step_size', OptimizerDefaultValues.CYCLIC_STEP_SIZE),
                )
            elif lr_schedule_type == CyclicLearningRateType.EXPONENTIAL:
                learning_rate_schedule = cyclic_learning_rate.ExponentialCyclicalLearningRate(
                    initial_learning_rate=lr_schedule.get(
                        'initial_learning_rate', OptimizerDefaultValues.CYCLIC_INITIAL_LEARNING_RATE),
                    maximal_learning_rate=lr_schedule.get(
                        'maximal_learning_rate', OptimizerDefaultValues.CYCLIC_MAXIMAL_LEARNING_RATE),
                    step_size=lr_schedule.get(
                        'step_size', OptimizerDefaultValues.CYCLIC_STEP_SIZE),
                    gamma=lr_schedule.get(
                        'gamma', OptimizerDefaultValues.CYCLIC_GAMMA),
                )
            else:
                raise ValueError(
                    "Unsupported cyclic learning rate schedule type key: " + lr_schedule_type)
        else:
            raise ValueError(
                "Unsupported learning rate schedule key: " + lr_schedule_key)

    return learning_rate_schedule


def get_optimizer(model_config) -> tf_optimizers.Optimizer:
    """
    Define the optimizer used for training the RelevanceModel
    Users have the option to define an ExponentialDecay learning rate schedule

    Parameters
    ----------
    model_config_file : str
        Path to model config file

    Returns
    -------
    tensorflow keras optimizer

    Notes
    -----
    References:
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
        https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
        https://arxiv.org/pdf/1506.01186.pdf
    """
    learning_rate_schedule = choose_scheduler(model_config)
    return choose_optimizer(model_config, learning_rate_schedule)
