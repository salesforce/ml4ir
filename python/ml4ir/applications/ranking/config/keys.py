from ml4ir.base.config.keys import Key


class LossKey(Key):
    """Model loss keys"""

    CATEGORICAL_CROSS_ENTROPY = "categorical_cross_entropy"
    SIGMOID_CROSS_ENTROPY = "sigmoid_cross_entropy"
    RANK_ONE_LISTNET = "rank_one_listnet"


class ScoringTypeKey(Key):
    """Scoring keys"""

    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    GROUPWISE = "groupwise"
    LISTWISE = "listwise"


class MetricKey(Key):
    """Model metric keys"""

    MRR = "MRR"
    ACR = "ACR"
    NDCG = "NDCG"
    PRECISION = 'Precision'
    CATEGORICAL_ACCURACY = "categorical_accuracy"


class LossTypeKey(Key):
    """Loss type keys"""

    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    LISTWISE = "listwise"
