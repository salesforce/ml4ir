from ml4ir.base.config.keys import ArchitectureKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.model.architectures.dnn import DNN


def get_architecture(model_config: dict, feature_config: FeatureConfig, file_io):
    """
    Return the architecture operation based on the model_config YAML specified
    """
    architecture_key = model_config.get("architecture_key")
    if architecture_key == ArchitectureKey.DNN:
        return DNN(model_config, feature_config, file_io).get_architecture_op()

    elif architecture_key == ArchitectureKey.RNN:
        raise NotImplementedError

    else:
        raise NotImplementedError
