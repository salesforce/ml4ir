from tensorflow import saved_model


class Key:
    """Base class for Keys"""

    @classmethod
    def get_all_keys(cls):
        keys = list()
        for attr in dir(cls):
            if not callable(getattr(cls, attr)):
                if not attr.startswith("__"):
                    keys.append(cls.__dict__[attr])

        return keys


class ArchitectureKey(Key):
    """Model architecture keys"""

    DNN = "dnn"
    RNN = "rnn"


class LossKey(Key):
    """Model loss keys"""

    SIGMOID_CROSS_ENTROPY = "sigmoid_cross_entropy"
    RANK_ONE_LISTNET = "rank_one_listnet"


class ScoringKey(Key):
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
    CATEGORICAL_ACCURACY = "categorical_accuracy"


class LossTypeKey(Key):
    """Loss type keys"""

    POINTWISE = "pointwise"
    PAIRWISE = "pairwise"
    LISTWISE = "listwise"


class OptimizerKey(Key):
    """Model optimizer keys"""

    ADAM = "adam"
    ADAGRAD = "adagrad"
    NADAM = "nadam"
    SGD = "sgd"
    RMS_PROP = "rms_prop"


class DataFormatKey(Key):
    """Data Format keys"""

    CSV = "csv"
    TFRECORD = "tfrecord"


class DataSplitKey(Key):
    """Data Split keys"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class FeatureTypeKey(Key):
    """Feature Data Type keys"""

    NUMERIC = "numeric"
    STRING = "string"
    CATEGORICAL = "categorical"
    LABEL = "label"


class TFRecordTypeKey(Key):
    """Type of data in TFRecord SequenceExample protobuf"""

    SEQUENCE = "sequence"
    CONTEXT = "context"


class ExecutionModeKey(Key):
    """Type of execution mode for the pipeline"""

    TRAIN_EVALUATE = "train_evaluate"
    TRAIN_ONLY = "train_only"
    INFERENCE_ONLY = "inference_only"
    EVALUATE_ONLY = "evaluate_only"
    INFERENCE_EVALUATE = "inference_evaluate"
    TRAIN_INFERENCE_EVALUATE = "train_inference_evaluate"
    TRAIN_INFERENCE = "train_inference"


class ServingSignatureKey(Key):
    """Serving signature names"""

    DEFAULT = saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    TFRECORD = "serving_tfrecord"


class EmbeddingTypeKey(Key):
    """Types of embeddings"""

    BILSTM = "bilstm"
    CNN = "cnn"
