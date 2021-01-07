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
        string_tensor = [["abc"], ["xyz"], ["123"]]
        sequence_encoding = sequence_fns.bytes_sequence_to_encoding_bilstm(
            string_tensor, feature_info, self.file_io
        )

        # Assert the right shapes of the resulting encoding based on the feature_info
        assert sequence_encoding.shape[0] == len(string_tensor)
        assert sequence_encoding.shape[1] == 1
        assert sequence_encoding.shape[2] == encoding_size

    def test_categorical_embedding_to_encoding_bilstm(self):
        """
        Asserts the conversion of a string tensor to its corresponding sequence encoding
        obtained through the categorical_embedding_to_encoding_bilstm function
        Works by converting each string into a bytes sequence and then
        passing it through a biLSTM.

        The embedding and encoding dimensions are controlled by the feature_info
        """
        embedding_size = 128
        encoding_size = 512
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "type": "numeric",
                "fn": "categorical_embedding_to_encoding_bilstm",
                "args": {
                    "vocabulary_file": "ml4ir/applications/classification/tests/data/configs/vocabulary/entity_id.csv",
                    "encoding_type": "bilstm",
                    "encoding_size": encoding_size,
                    "embedding_size": embedding_size,
                },
            },
            "tfrecord_type": SequenceExampleTypeKey.CONTEXT,
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            [[["AAA"]], [["BBB"]], [["AAA"]], [["CCC"]], [["out_of_vocabulary"]]]
        )
        sequence_encoding = categorical_fns.categorical_embedding_to_encoding_bilstm(
            string_tensor, feature_info, self.file_io
        )

        # Assert the right shapes of the resulting encoding based on the feature_info
        assert sequence_encoding.shape[0] == len(string_tensor)
        assert sequence_encoding.shape[1] == 1
        assert sequence_encoding.shape[2] == encoding_size

        # Strings 0 and 2 should result in the same embedding because they are the same
        assert tf.reduce_all(tf.equal(sequence_encoding[0], sequence_encoding[2]))
        assert not tf.reduce_all(tf.equal(sequence_encoding[0], sequence_encoding[1]))
        assert not tf.reduce_all(tf.equal(sequence_encoding[3], sequence_encoding[4]))
        assert not tf.reduce_all(tf.equal(sequence_encoding[1], sequence_encoding[4]))

    def test_categorical_embedding_to_encoding_bilstm_file_truncation(self):
        """
        Asserts the conversion of a string tensor to its corresponding sequence encoding
        obtained through the categorical_embedding_to_encoding_bilstm function
        Works by converting each string into a bytes sequence and then
        passing it through a biLSTM.

        The embedding and encoding dimensions are controlled by the feature_info
        """
        max_length = 2
        embedding_size = 128
        encoding_size = 512
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "type": "numeric",
                "fn": "categorical_embedding_to_encoding_bilstm",
                "args": {
                    "vocabulary_file": "ml4ir/applications/classification/tests/data/configs/vocabulary/entity_id.csv",
                    "encoding_type": "bilstm",
                    "encoding_size": encoding_size,
                    "embedding_size": embedding_size,
                    "max_length": max_length,
                },
            },
            "tfrecord_type": SequenceExampleTypeKey.CONTEXT,
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            [[["AAA"]], [["BBB"]], [["AAA"]], [["CCC"]], [["out_of_vocabulary"]]]
        )
        sequence_encoding = categorical_fns.categorical_embedding_to_encoding_bilstm(
            string_tensor, feature_info, self.file_io
        )

        # Assert the right shapes of the resulting encoding based on the feature_info
        assert sequence_encoding.shape[0] == len(string_tensor)
        assert sequence_encoding.shape[1] == 1
        assert sequence_encoding.shape[2] == encoding_size

        assert tf.reduce_all(tf.equal(sequence_encoding[0], sequence_encoding[2]))
        assert not tf.reduce_all(tf.equal(sequence_encoding[0], sequence_encoding[1]))
        assert tf.reduce_all(tf.equal(sequence_encoding[3], sequence_encoding[4]))
        assert not tf.reduce_all(tf.equal(sequence_encoding[1], sequence_encoding[4]))

    def test_categorical_embedding_to_encoding_bilstm_sequence_of_words(self):
        """
        Check that categorical_embedding_to_encoding_bilstm function does not fail with sequences of word (in a single
        row).
        """
        embedding_size = 32
        encoding_size = 64
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "type": "numeric",
                "fn": "categorical_embedding_to_encoding_bilstm",
                "args": {
                    "vocabulary_file": "ml4ir/applications/classification/tests/data/configs/vocabulary/entity_id.csv",
                    "encoding_type": "bilstm",
                    "encoding_size": encoding_size,
                    "embedding_size": embedding_size,
                    "dropout_rate": 0.2,
                },
            },
            "tfrecord_type": SequenceExampleTypeKey.CONTEXT,
        }

        string_tensor = tf.constant([[["AAA", "BBB", "out_of_vocabulary"]]])

        categorical_fns.categorical_embedding_to_encoding_bilstm(
            string_tensor, feature_info, self.file_io
        )

    def test_categorical_embedding_to_encoding_bilstm_oov_mapping_with_dropout(self):
        """
        Asserts "out of vocabulary" words are mapped to the same value when dropout_rate is used,
        while word "in vocabulary" are mapped to their own embeddings.
        """
        embedding_size = 128
        encoding_size = 512
        vocabulary_file = (
            "ml4ir/applications/classification/tests/data/configs/vocabulary/entity_id.csv"
        )

        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "type": "numeric",
                "fn": "categorical_embedding_to_encoding_bilstm",
                "args": {
                    "vocabulary_file": vocabulary_file,
                    "encoding_type": "bilstm",
                    "encoding_size": encoding_size,
                    "embedding_size": embedding_size,
                    "dropout_rate": 0.2,
                },
            },
            "tfrecord_type": SequenceExampleTypeKey.CONTEXT,
        }

        df_vocabulary = self.file_io.read_df(vocabulary_file)
        vocabulary = df_vocabulary.values.flatten().tolist()

        # Define a batch where each row is a single word from the vocabulary (so that we can later compare each word
        # encoding)
        words = vocabulary + ["out_of_vocabulary", "out_of_vocabulary2"]
        string_tensor = tf.constant([[[word]] for word in words])

        # If the embedding lookup input dimension were computed incorrectly, the following call would fail with:
        # tensorflow.python.framework.errors_impl.InvalidArgumentError: indices[0,0,7] = 8 is not in [0, 8)
        sequence_encoding = categorical_fns.categorical_embedding_to_encoding_bilstm(
            string_tensor, feature_info, self.file_io
        )

        # Strings 1 and 2 should result in the same embedding because they are both OOV
        # Check that each word, except the OOV ones are mapped to different encodings (actually different embeddings,
        # but the encoding on top should be equivalent as each row is a single word).
        for word1_position in range(0, len(vocabulary)):
            # +1 to include one OOV word
            for word2_position in range(word1_position + 1, len(vocabulary) + 1):
                if tf.reduce_all(
                    tf.equal(sequence_encoding[word1_position], sequence_encoding[word2_position])
                ):
                    word1, word2 = words[word1_position], words[word2_position]
                    self.assertFalse(
                        tf.reduce_all(
                            tf.equal(
                                sequence_encoding[word1_position],
                                sequence_encoding[word2_position],
                            )
                        ),
                        msg="Check that non-OOV words map to different encodings, and not also not to OOV:"
                        "{} vs. {}".format(word1, word2),
                    )

        self.assertTrue(
            tf.reduce_all(tf.equal(sequence_encoding[-1], sequence_encoding[-2])),
            msg="Check that OOV words map to the same encodings",
        )
        assert not tf.reduce_all(tf.equal(sequence_encoding[0], sequence_encoding[1]))

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
        string_tensor = ["domain_0", "domain_1", "domain_0"]

        categorical_embedding = categorical_fns.categorical_embedding_with_hash_buckets(
            string_tensor, feature_info, self.file_io
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
            index_tensor, feature_info, self.file_io
        )

        # Assert the right shapes of the resulting embedding
        assert categorical_embedding.shape[0] == len(index_tensor)
        assert categorical_embedding.shape[1] == 1
        assert categorical_embedding.shape[2] == embedding_size

        # Assert equality of embeddings with same indices
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[1]))
        assert tf.reduce_all(tf.equal(categorical_embedding[1], categorical_embedding[3]))
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[-1]))

    def test_categorical_embedding_with_vocabulary_file_with_ids(self):
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
                    "vocabulary_file": "ml4ir/applications/ranking/tests/data/configs/domain_name_vocab.csv",
                    "embedding_size": embedding_size,
                    "default_value": -1,
                    "num_oov_buckets": 1,
                },
            },
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        categorical_embedding = categorical_fns.categorical_embedding_with_vocabulary_file(
            string_tensor, feature_info, self.file_io
        )

        # Assert the right shapes of the resulting embedding
        assert categorical_embedding.shape[0] == len(string_tensor)
        assert categorical_embedding.shape[1] == 1
        assert categorical_embedding.shape[2] == embedding_size

        # Strings 0 and 2 should result in the same embedding because they are the same
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[2]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[1]))
        assert tf.reduce_all(tf.equal(categorical_embedding[4], categorical_embedding[5]))

        # Strings domain_0 and domain_2 should result in the same embedding because they are mapped to the same ID
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[3]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[3], categorical_embedding[4]))

    def test_categorical_embedding_with_vocabulary_file_without_ids(self):
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
                    "default_value": -1,
                    "num_oov_buckets": 1,
                },
            },
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        categorical_embedding = categorical_fns.categorical_embedding_with_vocabulary_file(
            string_tensor, feature_info, self.file_io
        )

        # Assert the right shapes of the resulting embedding
        assert categorical_embedding.shape[0] == len(string_tensor)
        assert categorical_embedding.shape[1] == 1
        assert categorical_embedding.shape[2] == embedding_size

        # Strings 0 and 2 should result in the same embedding because they are the same
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[2]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[1]))
        assert tf.reduce_all(tf.equal(categorical_embedding[4], categorical_embedding[5]))

        # Strings domain_0 and domain_2 should NOT result in the same embedding because they use a default one-to-one vocabulary mapping
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[3]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[3], categorical_embedding[4]))

    def test_categorical_embedding_with_vocabulary_file_with_ids_and_dropout(self):
        """
        Asserts the conversion of a categorical string tensor into an embedding representation
        Works by converting the string into indices using a vocabulary file and then dropping these indices into the OOV index at dropout_rate rate.

        The embedding size, dropout_rate are controlled by feature_info
        """
        embedding_size = 4
        dropout_rate = 0.999
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "categorical_embedding_with_vocabulary_file",
                "args": {
                    "vocabulary_file": "ml4ir/applications/ranking/tests/data/configs/domain_name_vocab.csv",
                    "embedding_size": embedding_size,
                    "dropout_rate": dropout_rate,
                },
            },
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        value_error_raised = False
        try:
            categorical_fns.categorical_embedding_with_vocabulary_file_and_dropout(
                string_tensor, feature_info, self.file_io
            )
        except ValueError:
            # Should throw error as method does not work with IDs containing 0
            value_error_raised = True

        assert value_error_raised

    def test_categorical_embedding_with_vocabulary_file_without_ids_and_dropout(self):
        """
        Asserts the conversion of a categorical string tensor into an embedding representation
        Works by converting the string into indices using a vocabulary file and then dropping these indices into the OOV index at dropout_rate rate.

        The embedding size, dropout_rate are controlled by feature_info
        """
        embedding_size = 4
        dropout_rate = 0.999
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "categorical_embedding_with_vocabulary_file",
                "args": {
                    "vocabulary_file": "ml4ir/applications/ranking/tests/data/configs/domain_name_vocab_no_id.csv",
                    "embedding_size": embedding_size,
                    "dropout_rate": dropout_rate,
                },
            },
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        categorcial_tensor = tf.keras.Input(shape=(1,), dtype=tf.string)
        embedding_tensor = categorical_fns.categorical_embedding_with_vocabulary_file_and_dropout(
            categorcial_tensor, feature_info, self.file_io
        )
        model = tf.keras.Model(categorcial_tensor, embedding_tensor)

        categorical_embedding = model(string_tensor, training=False)

        # Assert the right shapes of the resulting embedding
        assert categorical_embedding.shape[0] == len(string_tensor)
        assert categorical_embedding.shape[1] == 1
        assert categorical_embedding.shape[2] == embedding_size

        # Strings 0 and 2 should result in the same embedding because they are the same
        assert tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[2]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[1]))
        assert tf.reduce_all(tf.equal(categorical_embedding[4], categorical_embedding[5]))

        # Strings domain_0 and domain_2 should NOT result in the same embedding because they use a default one-to-one vocabulary mapping
        assert not tf.reduce_all(tf.equal(categorical_embedding[0], categorical_embedding[3]))
        assert not tf.reduce_all(tf.equal(categorical_embedding[3], categorical_embedding[4]))

        categorical_embedding = model(string_tensor, training=True)
        # Since dropout_rate is set to 0.999, all categorical indices
        # should be masked to OOV index and thus the embeddings should be the same
        assert tf.reduce_all(tf.equal(categorical_embedding, categorical_embedding[0]))

    def test_categorical_indicator_with_vocabulary_file_with_ids(self):
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
                    "vocabulary_file": "ml4ir/applications/ranking/tests/data/configs/domain_name_vocab.csv",
                    "num_oov_buckets": 1,
                },
            },
            "default_value": "",
        }

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        categorical_one_hot = categorical_fns.categorical_indicator_with_vocabulary_file(
            string_tensor, feature_info, self.file_io
        )

        # Assert the right shapes of the resulting one-hot vector
        assert categorical_one_hot.shape[0] == len(string_tensor)
        assert categorical_one_hot.shape[1] == 1
        assert categorical_one_hot.shape[2] == 6
        assert tf.reduce_all(tf.squeeze(tf.reduce_sum(categorical_one_hot, axis=2)) == 1.0)

        # Strings 0 and 2 should result in the same one-hot vector because they are the same
        assert tf.reduce_all(tf.equal(categorical_one_hot[0], categorical_one_hot[2]))
        assert not tf.reduce_all(tf.equal(categorical_one_hot[0], categorical_one_hot[1]))
        assert tf.reduce_all(tf.equal(categorical_one_hot[4], categorical_one_hot[5]))

        # Strings domain_0 and domain_2 should result in the same one-hot vector because they are mapped to the same ID
        assert tf.reduce_all(tf.equal(categorical_one_hot[0], categorical_one_hot[3]))
        assert not tf.reduce_all(tf.equal(categorical_one_hot[3], categorical_one_hot[4]))

    def test_categorical_indicator_with_vocabulary_file_without_ids(self):
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

        categorical_one_hot = categorical_fns.categorical_indicator_with_vocabulary_file(
            string_tensor, feature_info, self.file_io
        )

        # Assert the right shapes of the resulting one-hot vector
        assert categorical_one_hot.shape[0] == len(string_tensor)
        assert categorical_one_hot.shape[1] == 1
        assert categorical_one_hot.shape[2] == 6
        assert tf.reduce_all(tf.squeeze(tf.reduce_sum(categorical_one_hot, axis=2)) == 1.0)

        # Strings 0 and 2 should result in the same one-hot vector because they are the same
        assert tf.reduce_all(tf.equal(categorical_one_hot[0], categorical_one_hot[2]))
        assert not tf.reduce_all(tf.equal(categorical_one_hot[0], categorical_one_hot[1]))
        assert tf.reduce_all(tf.equal(categorical_one_hot[4], categorical_one_hot[5]))

        # Strings domain_0 and domain_2 should NOT result in the same one-hot vector because they use a default one-to-one vocabulary mapping
        assert not tf.reduce_all(tf.equal(categorical_one_hot[0], categorical_one_hot[3]))
        assert not tf.reduce_all(tf.equal(categorical_one_hot[3], categorical_one_hot[4]))

    def test_global_1d_pooling(self):
        """
        Unit test the global 1D pooling feature function on sequence features

        Checks the right output shapes produced and the values generated
        """
        feature_tensor = tf.reshape(tf.constant(range(30), dtype=tf.float32), (2, 5, 3))
        pooled_tensor = sequence_fns.global_1d_pooling(
            feature_tensor=feature_tensor,
            feature_info={
                "name": "f",
                "feature_layer_info": {
                    "args": {"fns": ["sum", "mean", "max", "min", "count_nonzero"]}
                },
            },
            file_io=None,
        )

        assert pooled_tensor.shape == (2, 5, 5)
        assert (
            pooled_tensor.numpy()
            == [
                [
                    [3.0, 1.0, 2.0, 0.0, 2.0],
                    [12.0, 4.0, 5.0, 3.0, 3.0],
                    [21.0, 7.0, 8.0, 6.0, 3.0],
                    [30.0, 10.0, 11.0, 9.0, 3.0],
                    [39.0, 13.0, 14.0, 12.0, 3.0],
                ],
                [
                    [48.0, 16.0, 17.0, 15.0, 3.0],
                    [57.0, 19.0, 20.0, 18.0, 3.0],
                    [66.0, 22.0, 23.0, 21.0, 3.0],
                    [75.0, 25.0, 26.0, 24.0, 3.0],
                    [84.0, 28.0, 29.0, 27.0, 3.0],
                ],
            ]
        ).all()

        # Test function with padded values
        # Here, we mask all the even values
        padded_val = -1.0
        feature_tensor_with_mask = tf.where(feature_tensor % 2 == 0, feature_tensor, padded_val)
        pooled_tensor_with_mask = sequence_fns.global_1d_pooling(
            feature_tensor=feature_tensor_with_mask,
            feature_info={
                "name": "f",
                "feature_layer_info": {
                    "args": {
                        "fns": ["sum", "mean", "max", "min", "count_nonzero"],
                        "padded_val": padded_val,
                        "masked_max_val": 100.0,
                    }
                },
            },
            file_io=None,
        )
        assert pooled_tensor_with_mask.shape == (2, 5, 5)
        assert not (pooled_tensor_with_mask.numpy() == pooled_tensor.numpy()).all()
        assert (
            pooled_tensor_with_mask.numpy()
            == [
                [
                    [2.0, 1.0, 2.0, 0.0, 1.0],
                    [4.0, 4.0, 4.0, 4.0, 1.0],
                    [14.0, 7.0, 8.0, 6.0, 2.0],
                    [10.0, 10.0, 10.0, 10.0, 1.0],
                    [26.0, 13.0, 14.0, 12.0, 2.0],
                ],
                [
                    [16.0, 16.0, 16.0, 16.0, 1.0],
                    [38.0, 19.0, 20.0, 18.0, 2.0],
                    [22.0, 22.0, 22.0, 22.0, 1.0],
                    [50.0, 25.0, 26.0, 24.0, 2.0],
                    [28.0, 28.0, 28.0, 28.0, 1.0],
                ],
            ]
        ).all()

        # Test empty pooling fn list
        found_value_error = False
        try:
            pooled_tensor = sequence_fns.global_1d_pooling(
                feature_tensor=feature_tensor,
                feature_info={
                    "name": "f",
                    "feature_layer_info": {"args": {"fns": [], "padded_val": padded_val}},
                },
                file_io=None,
            )
        except ValueError:
            found_value_error = True
        assert found_value_error

        # Test invalid pooling fn
        found_key_error = False
        try:
            pooled_tensor = sequence_fns.global_1d_pooling(
                feature_tensor=feature_tensor,
                feature_info={
                    "name": "f",
                    "feature_layer_info": {"args": {"fns": ["invalid"], "padded_val": padded_val}},
                },
                file_io=None,
            )
        except KeyError:
            found_key_error = True
        assert found_key_error
