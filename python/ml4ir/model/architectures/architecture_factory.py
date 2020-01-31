from ml4ir.config.keys import ArchitectureKey
from ml4ir.model.architectures import dnn


def get_architecture(architecture_key: str):
    """
    Return the architecture operation based on the key specified
    """

    if architecture_key == ArchitectureKey.SIMPLE_DNN:
        return dnn.get_architecture_op()

    elif architecture_key == ArchitectureKey.DNN_128:
        return dnn.get_architecture_op(128)

    elif architecture_key == ArchitectureKey.LSTM:
        raise NotImplementedError

    else:
        raise NotImplementedError
