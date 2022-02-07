import tensorflow as tf
from tensorflow.keras import layers

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import SequenceExampleTypeKey
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.features.feature_fns.sequence import BytesSequenceToEncodingBiLSTM
from ml4ir.base.features.feature_fns.sequence import Global1dPooling
from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingToEncodingBiLSTM
from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingWithHashBuckets
from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingWithIndices
from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingWithVocabularyFile
from ml4ir.base.features.feature_fns.categorical import CategoricalIndicatorWithVocabularyFile
from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingWithVocabularyFileAndDropout
from ml4ir.base.features.feature_fns.tf_native import TFNativeOpLayer
from ml4ir.base.io.file_io import FileIO


class FeatureLayerMap:
    """Class defining mapping from keys to feature layer functions"""

    def __init__(self):
        """
        Define ml4ir's predefined feature transformation functions
        """
        self.key_to_fn = {
            BytesSequenceToEncodingBiLSTM.__name__: BytesSequenceToEncodingBiLSTM,
            Global1dPooling.__name__: Global1dPooling,
            CategoricalEmbeddingToEncodingBiLSTM.__name__: CategoricalEmbeddingToEncodingBiLSTM,
            CategoricalEmbeddingWithHashBuckets.__name__: CategoricalEmbeddingWithHashBuckets,
            CategoricalEmbeddingWithIndices.__name__: CategoricalEmbeddingWithIndices,
            CategoricalEmbeddingWithVocabularyFile.__name__: CategoricalEmbeddingWithVocabularyFile,
            CategoricalEmbeddingWithVocabularyFileAndDropout.__name__: CategoricalEmbeddingWithVocabularyFileAndDropout,
            CategoricalIndicatorWithVocabularyFile.__name__: CategoricalIndicatorWithVocabularyFile,
            TFNativeOpLayer.__name__: TFNativeOpLayer
        }

    def add_fn(self, key, fn):
        """
        Add custom new function to the FeatureLayerMap

        Parameters
        ----------
        key : str
            name of the feature transformation function
        fn : tf.function
            tensorflow function that transforms input features
        """

        self.key_to_fn[key] = fn

    def add_fns(self, keys_to_fns_dict):
        """
        Add custom new functions to the FeatureLayerMap

        Parameters
        ----------
        keykeys_to_fns_dict : dict
            Dictionary with name and definition of custom
            tensorflow functions that transform input features
        """
        self.key_to_fn.update(keys_to_fns_dict)

    def get_fns(self):
        """
        Get all feature transformation functions

        Returns
        -------
        dict
            Dictionary of feature transformation functions
        """
        return self.key_to_fn

    def get_fn(self, key):
        """
        Get feature transformation function using the key

        Parameters
        ----------
        key : str
            Name of the feature transformation function to be fetched

        Returns
        -------
        tf.function
            Feature transformation function
        """
        return self.key_to_fn.get(key)

    def pop_fn(self, key):
        """
        Get feature transformation function using the key and remove
        from FeatureLayerMap

        Parameters
        ----------
        key : str
            Name of the feature transformation function to be fetched

        Returns
        -------
        tf.function
            Feature transformation function
        """
        self.key_to_fn.pop(key)
