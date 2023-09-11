from ml4ir.applications.ranking.features.feature_fns import categorical
from ml4ir.base.tests.test_base import RelevanceTestBase

import tensorflow as tf
import numpy as np


class FeatureLayerTest(RelevanceTestBase):

    def test_sequence_categorical_embedding_with_vocabulary_file_without_ids(self):
        """
        Asserts the conversion of a categorical string tensor into a categorical embedding
        Works by converting the string into indices using a vocabulary file and then
        converting the indices into categorical embeddings

        The embedding dimensions, buckets, etc are controlled by the feature_info
        """
        embedding_size = 32
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "categorical_embedding_with_vocabulary_file",
                "args": {
                    "vocabulary_file": "ml4ir/applications/ranking/tests/data/configs/domain_name_vocab_no_id.csv",
                    "embedding_size": embedding_size,
                    "num_oov_buckets": 1,
                },
            },
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        categorical_embedding = categorical.SequenceCategoricalEmbeddingWithVocabularyFile(
            feature_info, self.file_io
        )(string_tensor)

        # Assert the right shapes of the resulting embedding
        assert categorical_embedding.shape[0] == len(string_tensor)
        assert categorical_embedding.shape[1] == embedding_size

        # Strings 0 and 2 should result in the same embedding because they are the same
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[2]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[1]))
        assert tf.reduce_all(tf.equal(categorical_embedding[4], categorical_embedding[5]))

        # Strings domain_0 and domain_2 should NOT result in the same embedding because they use a default one-to-one vocabulary mapping
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[3]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[3], categorical_embedding[4]))

    def test_sequence_categorical_indicator_with_vocabulary_file_without_ids(self):
        """
        Asserts the conversion of a categorical string tensor into a one-hot representation
        Works by converting the string into indices using a vocabulary file and then
        converting the indices into one-hot vectors

        The one-hot vector dimensions, buckets, etc are controlled by the feature_info
        """
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "categorical_indicator_with_vocabulary_file",
                "args": {
                    "vocabulary_file": "ml4ir/applications/ranking/tests/data/configs/domain_name_vocab_no_id.csv",
                    "num_oov_buckets": 1,
                },
            },
            "default_value": "",
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        categorical_one_hot = categorical.SequenceCategoricalIndicatorWithVocabularyFile(
            feature_info, self.file_io
        )(string_tensor)

        # Assert the right shapes of the resulting one-hot vector
        assert categorical_one_hot.shape[0] == len(string_tensor)
        assert categorical_one_hot.shape[1] == 6
        assert tf.reduce_all(tf.squeeze(tf.reduce_sum(categorical_one_hot, axis=1)) == 1.0)

        # Strings 0 and 2 should result in the same one-hot vector because they are the same
        assert tf.reduce_all(tf.equal(categorical_one_hot[0], categorical_one_hot[2]))
        assert not tf.reduce_all(tf.equal(categorical_one_hot[0], categorical_one_hot[1]))
        assert tf.reduce_all(tf.equal(categorical_one_hot[4], categorical_one_hot[5]))

        # Strings domain_0 and domain_2 should NOT result in the same one-hot vector because they use a default one-to-one vocabulary mapping
        assert not tf.reduce_all(tf.equal(categorical_one_hot[0], categorical_one_hot[3]))
        assert not tf.reduce_all(tf.equal(categorical_one_hot[3], categorical_one_hot[4]))