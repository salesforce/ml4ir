from ml4ir.base.config.keys import Key


class LossKey(Key):
    """Model loss keys that can be used with a ranking model"""

    CATEGORICAL_CROSS_ENTROPY = "categorical_cross_entropy"
    SIGMOID_CROSS_ENTROPY = "sigmoid_cross_entropy"
    RANK_ONE_LISTNET = "rank_one_listnet"
    SOFTMAX_CROSS_ENTROPY = "softmax_cross_entropy"
    BASIC_CROSS_ENTROPY = "basic_cross_entropy"


class ScoringTypeKey(Key):
    """Scoring keys"""

    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    GROUPWISE = "groupwise"
    LISTWISE = "listwise"


class MetricKey(Key):
    """Model metric keys that can be used with a ranking model"""

    MRR = "MRR"
    ACR = "ACR"
    NDCG = "NDCG"
    PRECISION = "Precision"
    RankMatchFailure = "RankMatchFailure"
    CATEGORICAL_ACCURACY = "categorical_accuracy"
    TOP_5_CATEGORICAL_ACCURACY = "top_5_categorical_accuracy"


class LossTypeKey(Key):
    """Loss type keys"""

    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    LISTWISE = "listwise"


class PositionalBiasHandler(Key):
    """Types of models to handle positional bias"""

    FIXED_ADDITIVE_POSITIONAL_BIAS = "fixed_additive_positional_bias"
