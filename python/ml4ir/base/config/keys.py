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
    LINEAR = "linear"
    RNN = "rnn"
    AUTO_DAG_NETWORK = "auto-dag-network"


class OptimizerKey(Key):
    """Model optimizer keys"""

    ADAM = "adam"
    ADAGRAD = "adagrad"
    NADAM = "nadam"
    SGD = "sgd"
    RMS_PROP = "rms_prop"

class LearningRateScheduleKey(Key):
    """Learning rate schedule keys"""

    EXPONENTIAL = "exponential"
    CYCLIC = "cyclic"
    CONSTANT = 'constant'
    REDUCE_LR_ON_PLATEAU = 'reduce_lr_on_plateau'

class CyclicLearningRateType(Key):
    """Cyclic learning rate schedule type keys"""

    TRIANGULAR = "triangular"
    TRIANGULAR2 = "triangular2"
    EXPONENTIAL = "exponential"


class DataFormatKey(Key):
    """Data Format keys"""

    CSV = "csv"
    TFRECORD = "tfrecord"
    RANKLIB = "ranklib"


class DataSplitKey(Key):
    """Data Split keys"""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


class TFRecordTypeKey(Key):
    """Example or SequenceExample"""

    EXAMPLE = "example"
    SEQUENCE_EXAMPLE = "sequence_example"


class SequenceExampleTypeKey(Key):
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
    INFERENCE_RESAVE = "inference_resave"
    EVALUATE_RESAVE = "evaluate_resave"
    INFERENCE_EVALUATE_RESAVE = "inference_evaluate_resave"
    RESAVE_ONLY = "resave_only"


class ServingSignatureKey(Key):
    """Serving signature names"""

    DEFAULT = saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY
    TFRECORD = "serving_tfrecord"


class EncodingTypeKey(Key):
    """Types of embeddings"""

    BILSTM = "bilstm"
    CNN = "cnn"


class DefaultDirectoryKey(Key):
    """Default directory paths"""

    MODELS = "models"
    LOGS = "logs"
    DATA = "data"
    TEMP_DATA = "data/temp"
    TEMP_MODELS = "models/temp"


class FileHandlerKey(Key):
    """File handler type"""

    LOCAL = "local"
    SPARK = "spark"


class CalibrationKey(Key):
    CALIBRATION = "calibration"
    TEMPERATURE_SCALING = "temperature_scaling"
    TEMPERATURE = "temperature"
    ARGS = "args"


class PipelineType(Key):
    RANKING_PIPELINE = "RankingPipeline"
    CLASSIFICATION_PIPELINE = "ClassificationPipeline"


class FeatureTypeKey(Key):
    """Type of feature"""
    TRAIN = "train"  # Features used as inputs to the network architecture
    METADATA = "metadata"  # Features used as metadata to learn the weights
    MASK = "mask"
    LOGITS = "logits"
    SCORES = "scores"
    AUX_LABEL = "aux_label"


class VocabularyInfoArgsKey(Key):
    """Argument values for fetching vocabulary from a file using get_vocabulary_info()"""
    VOCABULARY_FILE = "vocabulary_file"
    KEY = "key"
    ID = "id"
    DEFAULT_VALUE = "default_value"
    MAX_LENGTH = "max_length"
    DROPOUT_RATE = "dropout_rate"