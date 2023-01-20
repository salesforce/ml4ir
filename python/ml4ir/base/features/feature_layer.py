from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingToEncodingBiLSTM
from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingWithHashBuckets
from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingWithIndices
from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingWithVocabularyFile
from ml4ir.base.features.feature_fns.categorical import CategoricalEmbeddingWithVocabularyFileAndDropout
from ml4ir.base.features.feature_fns.categorical import CategoricalIndicatorWithVocabularyFile
from ml4ir.base.features.feature_fns.sequence import BytesSequenceToEncodingBiLSTM
from ml4ir.base.features.feature_fns.sequence import Global1dPooling
from ml4ir.base.features.feature_fns.tf_native import TFNativeOpLayer


class FeatureLayerMap:
    """Class defining mapping from keys to feature layer functions"""

    def __init__(self):
        """
        Define ml4ir's predefined feature transformation functions
        """
        self.key_to_fn = {
            BytesSequenceToEncodingBiLSTM.LAYER_NAME: BytesSequenceToEncodingBiLSTM,
            Global1dPooling.LAYER_NAME: Global1dPooling,
            CategoricalEmbeddingToEncodingBiLSTM.LAYER_NAME: CategoricalEmbeddingToEncodingBiLSTM,
            CategoricalEmbeddingWithHashBuckets.LAYER_NAME: CategoricalEmbeddingWithHashBuckets,
            CategoricalEmbeddingWithIndices.LAYER_NAME: CategoricalEmbeddingWithIndices,
            CategoricalEmbeddingWithVocabularyFile.LAYER_NAME: CategoricalEmbeddingWithVocabularyFile,
            CategoricalEmbeddingWithVocabularyFileAndDropout.LAYER_NAME: CategoricalEmbeddingWithVocabularyFileAndDropout,
            CategoricalIndicatorWithVocabularyFile.LAYER_NAME: CategoricalIndicatorWithVocabularyFile,
            TFNativeOpLayer.LAYER_NAME: TFNativeOpLayer
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