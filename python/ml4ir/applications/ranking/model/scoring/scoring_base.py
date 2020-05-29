from ml4ir.base.model.architectures import architecture_factory


class ScoringBase(object):
    def __init__(self, scoring_key, model_config, loss_type):
        self.scoring_key = scoring_key
        self.model_config = model_config
        self.loss_type = loss_type

        self.architecture_op = architecture_factory.get_architecture(model_config=model_config)

    def get_scoring_fn(self, **kwargs):
        """
        Define a ranking scoring function with specified architecture
        """

        def _scoring_fn(features):
            pass

        return _scoring_fn
