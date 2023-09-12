import tensorflow as tf
import numpy as np

from ml4ir.applications.ranking.features.feature_fns import categorical
from ml4ir.base.tests.test_base import RelevanceTestBase


class FeatureLayerTest(RelevanceTestBase):

    def test_sequence_categorical_vector_embedding(self):
        """
        Asserts the conversion of a categorical string tensor into a categorical embedding
        Works by converting the string into indices using a vocabulary file and then
        converting the indices into categorical embeddings

        The embedding dimensions, buckets, etc are controlled by the feature_info
        """
        embedding_weights = np.random.randn(5 + 1, 32)
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "sequence_categorical_vector",
                "args": {
                    "vocabulary_file": "ml4ir/applications/ranking/tests/data/configs/domain_name_vocab_no_id.csv",
                    "embedding_size": 32,
                    "num_oov_buckets": 1,
                    "output_mode": "embedding",
                    "embeddings_initializer": tf.keras.initializers.Constant(embedding_weights)
                },
            },
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        actual_embedding = categorical.SequenceCategoricalVector(
            feature_info, self.file_io
        )(string_tensor).numpy()

        # Check if the embedding vectors match what we expect from the preset embedding weights
        # NOTE - Out of vocabulary tokens are mapped to 0 index
        expected_embedding = np.stack([embedding_weights[i] for i in [1, 2, 1, 3, 0, 0]])
        self.assertTrue(np.isclose(actual_embedding, expected_embedding).all())

    def test_sequence_categorical_vector_one_hot(self):
        """
        Asserts the conversion of a categorical string tensor into a one-hot representation
        Works by converting the string into indices using a vocabulary file and then
        converting the indices into one-hot vectors

        The one-hot vector dimensions, buckets, etc are controlled by the feature_info
        """
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "sequence_categorical_vector",
                "args": {
                    "vocabulary_file": "ml4ir/applications/ranking/tests/data/configs/domain_name_vocab_no_id.csv",
                    "num_oov_buckets": 1,
                    "output_mode": "one_hot"
                },
            },
            "default_value": "",
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        actual_one_hot = categorical.SequenceCategoricalVector(
            feature_info, self.file_io
        )(string_tensor).numpy()

        # Check if the one hot vectors match what we expect
        # NOTE - Out of vocabulary tokens are mapped to 0 index
        expected_one_hot = np.eye(6)[[1, 2, 1, 3, 0, 0]]
        self.assertTrue(np.isclose(actual_one_hot, expected_one_hot).all())