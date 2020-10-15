import tensorflow.keras.optimizers as tf_optimizers
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from ml4ir.base.model import cyclic_learning_rate
from ml4ir.base.config.keys import OptimizerKey, LearningRateScheduleKey, CyclicLearningRateType



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

    if 'lr_schedule' not in model_config:
        learning_rate_schedule = ExponentialDecay(
            initial_learning_rate=0.01,
            decay_steps=10000000,
            decay_rate=1.0,
            staircase=True,
        )

    else:
        lr_schedule = model_config['lr_schedule']
        lr_schedule_key = lr_schedule['key']
        if lr_schedule_key == LearningRateScheduleKey.EXPONENTIAL:
            learning_rate_schedule = ExponentialDecay(
                initial_learning_rate=lr_schedule['learning_rate'] if 'learning_rate' in lr_schedule else 0.01,
                decay_steps=lr_schedule['learning_rate_decay_steps'] if 'learning_rate_decay_steps' in lr_schedule else 10000000,
                decay_rate=lr_schedule['learning_rate_decay'] if 'learning_rate_decay' in lr_schedule else 1.0,
                staircase=True,
            )

        elif lr_schedule_key == LearningRateScheduleKey.CYCLIC:
            lr_schedule_type = lr_schedule['type']
            if lr_schedule_type == CyclicLearningRateType.TRIANGULAR:
                learning_rate_schedule = cyclic_learning_rate.TriangularCyclicalLearningRate(
                    initial_learning_rate=lr_schedule['initial_learning_rate'] if 'initial_learning_rate' in lr_schedule else 0.001,
                    maximal_learning_rate=lr_schedule['maximal_learning_rate'] if 'maximal_learning_rate' in lr_schedule else 0.01,
                    step_size=lr_schedule['step_size'] if 'step_size' in lr_schedule else 10,
                )
            elif lr_schedule_type == CyclicLearningRateType.TRIANGULAR2:
                learning_rate_schedule = cyclic_learning_rate.Triangular2CyclicalLearningRate(
                    initial_learning_rate=lr_schedule['initial_learning_rate'] if 'initial_learning_rate' in lr_schedule else 0.001,
                    maximal_learning_rate=lr_schedule['maximal_learning_rate'] if 'maximal_learning_rate' in lr_schedule else 0.01,
                    step_size=lr_schedule['step_size'] if 'step_size' in lr_schedule else 10,
                )
            elif lr_schedule_type == CyclicLearningRateType.EXPONENTIAL:
                learning_rate_schedule = cyclic_learning_rate.ExponentialCyclicalLearningRate(
                    initial_learning_rate=lr_schedule['initial_learning_rate'] if 'initial_learning_rate' in lr_schedule else 0.001,
                    maximal_learning_rate=lr_schedule['maximal_learning_rate'] if 'maximal_learning_rate' in lr_schedule else 0.01,
                    step_size=lr_schedule['step_size'] if 'step_size' in lr_schedule else 10,
                    gamma=lr_schedule['gamma'] if 'gamma' in lr_schedule else 1.0,
                )
            else:
                raise ValueError("illegal cyclic learning rate schedule type key: " + lr_schedule_type)
        else:
            raise ValueError("illegal learning rate schedule key: " + lr_schedule_key)

    if 'optimizer' not in model_config:
        return tf_optimizers.Adam(learning_rate=learning_rate_schedule, clipvalue=5.0)
    else:
        optimizer_key = model_config['optimizer']['key']
        gradient_clip_value = model_config['optimizer']['gradient_clip_value']
        if optimizer_key == OptimizerKey.ADAM:
            return tf_optimizers.Adam(
                learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value if 'gradient_clip_value' in model_config['optimizer'] else 5.0
            )
        elif optimizer_key == OptimizerKey.NADAM:
            return tf_optimizers.Nadam(
                learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value if 'gradient_clip_value' in model_config['optimizer'] else 5.0
            )
        elif optimizer_key == OptimizerKey.ADAGRAD:
            return tf_optimizers.Adagrad(
                learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value if 'gradient_clip_value' in model_config['optimizer'] else 5.0
            )
        elif optimizer_key == OptimizerKey.SGD:
            return tf_optimizers.SGD(
                learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value if 'gradient_clip_value' in model_config['optimizer'] else 5.0
            )
        elif optimizer_key == OptimizerKey.RMS_PROP:
            return tf_optimizers.RMSprop(
                learning_rate=learning_rate_schedule, clipvalue=gradient_clip_value if 'gradient_clip_value' in model_config['optimizer'] else 5.0
            )
        else:
            raise ValueError("illegal Optimizer key: " + optimizer_key)
