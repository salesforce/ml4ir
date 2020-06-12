from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.features.feature_fns import categorical as categorical_fns
from ml4ir.base.features.feature_fns import sequence as sequence_fns
from ml4ir.base.config.keys import SequenceExampleTypeKey

import tensorflow as tf


class RankingModelTest(RankingTestBase):
    def test_bytes_sequence_to_encoding_bilstm(self):
        """
        Asserts the conversion of a string tensor to its corresponding sequence encoding
        obtained through the bytes_sequence_to_encoding_bilstm function
        Works by converting each string into a bytes sequence and then
        passing it through a biLSTM.

        The embedding and encoding dimensions are controlled by the feature_info
        """
        max_length = 20
        embedding_size = 128
        encoding_size = 512
        feature_info = {
            "feature_layer_info": {
                "type": "numeric",
                "fn": "bytes_sequence_to_encoding_bilstm",
                "args": {
                    "encoding_type": "bilstm",
                    "encoding_size": encoding_size,
                    "embedding_size": embedding_size,
                    "max_length": max_length,
                },
            },
            "tfrecord_type": SequenceExampleTypeKey.CONTEXT,
        }

        # Define an input string tensor
        string_tensor = ["abc", "xyz", "123"]

        sequence_encoding = sequence_fns.bytes_sequence_to_encoding_bilstm(
            string_tensor, feature_info
        )

        # Assert the right shapes of the resulting encoding based on the feature_info
        assert sequence_encoding.shape[0] == len(string_tensor)
        assert sequence_encoding.shape[1] == 1
        assert sequence_encoding.shape[2] == encoding_size

    def test_categorical_embedding_with_hash_buckets(self):
        """
        Goal:
        Convert a categorical string tensor into a categorical embedding
        Works by converting the string into n integers by hashing and then
        converting the integers into n embeddings.

        The embedding dimensions, buckets, etc are controlled by the feature_info
        """
        num_hash_buckets = 4
        hash_bucket_size = 64
        embedding_size = 32
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "categorical_embedding_with_hash_buckets",
                "args": {
                    "num_hash_buckets": num_hash_buckets,
                    "hash_bucket_size": hash_bucket_size,
                    "embedding_size": embedding_size,
                    "merge_mode": "concat",
                },
            },
        }

        # Define an input string tensor
        string_tensor = ["group_0", "group_1", "group_0"]

        categorical_embedding = categorical_fns.categorical_embedding_with_hash_buckets(
            string_tensor, feature_info
        )

        # Assert the right shapes of the resulting embedding
        assert categorical_embedding.shape[0] == len(string_tensor)
        assert categorical_embedding.shape[1] == 1
        assert categorical_embedding.shape[2] == num_hash_buckets * embedding_size

        # Strings 0 and 2 should result in the same embedding because they are the same
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[2]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[1]))

    def test_categorical_embedding_with_indices(self):
        """
        Asserts the conversion of integer categorical indices tensor into categorical embeddings

        The embeddding dimensions are controlled by the feature_info
        """
        num_buckets = 8
        embedding_size = 64
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "categorical_embedding_with_indices",
                "args": {
                    "num_buckets": num_buckets,
                    "embedding_size": embedding_size,
                    "default_value": 0,
                },
            },
        }

        # Define an input int tensor
        index_tensor = [0, 1, 2, 1, 10]

        categorical_embedding = categorical_fns.categorical_embedding_with_indices(
            index_tensor, feature_info
        )

        # Assert the right shapes of the resulting embedding
        assert categorical_embedding.shape[0] == len(index_tensor)
        assert categorical_embedding.shape[1] == 1
        assert categorical_embedding.shape[2] == embedding_size

        # Assert equality of embeddings with same indices
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[1]))
        assert tf.reduce_all(tf.equal(categorical_embedding[1], categorical_embedding[3]))
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[-1]))

    def test_categorical_embedding_with_vocabulary_file(self):
        """
        Asserts the conversion of a categorical string tensor into a categorical embedding
        Works by converting the string into indices using a vocabulary file and then
        converting the indices into categorical embeddings

        The embedding dimensions, buckets, etc are controlled by the feature_info
        """
        #####################################################
        # Test for vocabulary file with ids mapping specified
        #####################################################
        embedding_size = 32
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "categorical_embedding_with_hash_buckets",
                "args": {
                    "vocabulary_file": "ml4ir/applications/ranking/tests/data/config/group_name_vocab.csv",
                    "embedding_size": embedding_size,
                    "default_value": -1,
                    "num_oov_buckets": 1,
                },
            },
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["group_0", "group_1", "group_0", "group_2", "group_10", "group_11"]
        )

        categorical_embedding = categorical_fns.categorical_embedding_with_vocabulary_file(
            string_tensor, feature_info
        )

        # Assert the right shapes of the resulting embedding
        assert categorical_embedding.shape[0] == len(string_tensor)
        assert categorical_embedding.shape[1] == 1
        assert categorical_embedding.shape[2] == embedding_size

        # Strings 0 and 2 should result in the same embedding because they are the same
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[2]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[1]))
        assert tf.reduce_all(tf.equal(categorical_embedding[4], categorical_embedding[5]))

        # Strings group_0 and group_2 should result in the same embedding because they are mapped to the same ID
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[3]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[3], categorical_embedding[4]))

        ########################################################
        # Test for vocabulary file with no ids mapping specified
        ########################################################
        feature_info["feature_layer_info"]["args"][
            "vocabulary_file"
        ] = "ml4ir/applications/ranking/tests/data/config/group_name_vocab_no_id.csv"

        categorical_embedding = categorical_fns.categorical_embedding_with_vocabulary_file(
            string_tensor, feature_info
        )

        # Assert the right shapes of the resulting embedding
        assert categorical_embedding.shape[0] == len(string_tensor)
        assert categorical_embedding.shape[1] == 1
        assert categorical_embedding.shape[2] == embedding_size

        # Strings 0 and 2 should result in the same embedding because they are the same
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[2]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[1]))
        assert tf.reduce_all(tf.equal(categorical_embedding[4], categorical_embedding[5]))

        # Strings group_0 and group_2 should NOT result in the same embedding because they use a default one-to-one vocabulary mapping
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[3]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[3], categorical_embedding[4]))
