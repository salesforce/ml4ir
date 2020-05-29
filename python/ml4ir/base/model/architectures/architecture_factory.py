<<<<<<< HEAD:python/ml4ir/model/architectures/architecture_factory.py
# type: ignore
# TODO: Fix typing

from ml4ir.config.keys import ArchitectureKey
from ml4ir.model.architectures.dnn import DNN
=======
from ml4ir.base.config.keys import ArchitectureKey
from ml4ir.base.model.architectures.dnn import DNN
>>>>>>> ba34ff34526926aa009ac7d5fb1babdf15be570b:python/ml4ir/base/model/architectures/architecture_factory.py


def get_architecture(model_config):
    """
    Return the architecture operation based on the model_config YAML specified
    """
    architecture_key = model_config.get("architecture_key")
    if architecture_key == ArchitectureKey.DNN:
        return DNN(model_config).get_architecture_op()

    elif architecture_key == ArchitectureKey.RNN:
        raise NotImplementedError

    else:
        raise NotImplementedError
