from ml4ir.base.model.scoring.scoring_base import ScoringBase


class PointwiseScoring(ScoringBase):
    def get_scoring_fn(self, **kwargs):
        """
        Define a pointwise ranking scoring function with specified architecture
        """

        def _scoring_fn(features):
            return self.architecture_op(features)

        return _scoring_fn
