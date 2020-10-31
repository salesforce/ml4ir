import tensorflow.keras.optimizers as tf_optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from ml4ir.base.model.optimizers import cyclic_learning_rate
from ml4ir.base.config.keys import OptimizerKey, LearningRateScheduleKey, CyclicLearningRateType
import tensorflow as tf

class DefaultValues(object):
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
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Optimizer
            https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay
            https://arxiv.org/pdf/1506.01186.pdf
    """

    if 'optimizer' not in model_config:
        return tf_optimizers.Adam(learning_rate=learning_rate_schedule, clipvalue=5.0)
    else:
        optimizer_key = model_config['optimizer']['key']
        if 'gradient_clip_value' in model_config['optimizer']:
            config = {'learning_rate': learning_rate_schedule, 'clipvalue': model_config['optimizer']['gradient_clip_value']}
        else:
            config = {'learning_rate': learning_rate_schedule}
        return tf.keras.optimizers.get({'class_name':optimizer_key, 'config':config})

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
        #use constant lr schedule
        learning_rate_schedule = ExponentialDecay(
            initial_learning_rate=DefaultValues.CONSTANT_LR,
            decay_steps=10000000,
            decay_rate=1.0,
        )

    else:
        lr_schedule = model_config['lr_schedule']
        lr_schedule_key = lr_schedule['key']

        if lr_schedule_key == LearningRateScheduleKey.EXPONENTIAL:
            learning_rate_schedule = ExponentialDecay(
                initial_learning_rate=lr_schedule['learning_rate'] if 'learning_rate' in lr_schedule else DefaultValues.CONSTANT_LR,
                decay_steps=lr_schedule['learning_rate_decay_steps'] if 'learning_rate_decay_steps' in lr_schedule else DefaultValues.EXP_DECAY_STEPS,
                decay_rate=lr_schedule['learning_rate_decay'] if 'learning_rate_decay' in lr_schedule else DefaultValues.EXP_DECAY_RATE,
                staircase=True,
            )

        elif lr_schedule_key == LearningRateScheduleKey.CONSTANT:
            learning_rate_schedule = ExponentialDecay(
                initial_learning_rate=lr_schedule['learning_rate'] if 'learning_rate' in lr_schedule else DefaultValues.CONSTANT_LR,
                decay_steps=10000000,
                decay_rate=1.0,
            )

        elif lr_schedule_key == LearningRateScheduleKey.CYCLIC:
            lr_schedule_type = lr_schedule['type']
            if lr_schedule_type == CyclicLearningRateType.TRIANGULAR:
                learning_rate_schedule = cyclic_learning_rate.TriangularCyclicalLearningRate(
                    initial_learning_rate=lr_schedule['initial_learning_rate'] if 'initial_learning_rate' in lr_schedule else DefaultValues.CYCLIC_INITIAL_LEARNING_RATE,
                    maximal_learning_rate=lr_schedule['maximal_learning_rate'] if 'maximal_learning_rate' in lr_schedule else DefaultValues.CYCLIC_MAXIMAL_LEARNING_RATE,
                    step_size=lr_schedule['step_size'] if 'step_size' in lr_schedule else DefaultValues.CYCLIC_STEP_SIZE,
                )
            elif lr_schedule_type == CyclicLearningRateType.TRIANGULAR2:
                learning_rate_schedule = cyclic_learning_rate.Triangular2CyclicalLearningRate(
                    initial_learning_rate=lr_schedule['initial_learning_rate'] if 'initial_learning_rate' in lr_schedule else DefaultValues.CYCLIC_INITIAL_LEARNING_RATE,
                    maximal_learning_rate=lr_schedule['maximal_learning_rate'] if 'maximal_learning_rate' in lr_schedule else DefaultValues.CYCLIC_MAXIMAL_LEARNING_RATE,
                    step_size=lr_schedule['step_size'] if 'step_size' in lr_schedule else DefaultValues.CYCLIC_STEP_SIZE,
                )
            elif lr_schedule_type == CyclicLearningRateType.EXPONENTIAL:
                learning_rate_schedule = cyclic_learning_rate.ExponentialCyclicalLearningRate(
                    initial_learning_rate=lr_schedule['initial_learning_rate'] if 'initial_learning_rate' in lr_schedule else DefaultValues.CYCLIC_INITIAL_LEARNING_RATE,
                    maximal_learning_rate=lr_schedule['maximal_learning_rate'] if 'maximal_learning_rate' in lr_schedule else DefaultValues.CYCLIC_MAXIMAL_LEARNING_RATE,
                    step_size=lr_schedule['step_size'] if 'step_size' in lr_schedule else DefaultValues.CYCLIC_STEP_SIZE,
                    gamma=lr_schedule['gamma'] if 'gamma' in lr_schedule else DefaultValues.CYCLIC_GAMMA,
                )
            else:
                raise ValueError("Unsupported cyclic learning rate schedule type key: " + lr_schedule_type)
        else:
            raise ValueError("Unsupported learning rate schedule key: " + lr_schedule_key)

    return learning_rate_schedule

def get_optimizer(file_io, model_config_file) -> tf_optimizers.Optimizer:
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

    FIXME:
        Define all arguments overriding tensorflow defaults in a separate file
        for visibility with ml4ir users
    """
    model_config = file_io.read_yaml(model_config_file)
    learning_rate_schedule = choose_scheduler(model_config)
    return choose_optimizer(model_config, learning_rate_schedule)


