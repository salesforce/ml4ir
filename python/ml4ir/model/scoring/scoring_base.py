from ml4ir.model.architectures import architecture_factory


class ScoringBase(object):
    def __init__(self, scoring_key, architecture_key, loss_type):
        self.scoring_key = scoring_key
        self.architecture_key = architecture_key
        self.loss_type = loss_type

        self.architecture_op = architecture_factory.get_architecture(
            architecture_key=architecture_key
        )

    def get_scoring_fn(self, **kwargs):
        """
        Define a ranking scoring function with specified architecture
        """

        def _scoring_fn(features):
            pass

        return _scoring_fn
