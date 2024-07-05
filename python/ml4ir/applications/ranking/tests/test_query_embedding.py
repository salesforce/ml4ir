import tensorflow as tf
import numpy as np
import unittest
from unittest.mock import MagicMock
from ml4ir.applications.ranking.features.feature_fns import string as string_transforms
from ml4ir.base.tests.test_base import RelevanceTestBase

class TestQueryEmbeddingVectorUsingGlove(RelevanceTestBase):
    def setUp(self):
        self.feature_info = {
            "name": "default_feature",
            "feature_layer_info": {
                "args": {
                    "embedding_size": 3,
                    "glove_path": "mock_glove_path.txt",
                    "max_entries": 2
                }
            }
        }
        self.file_io = MagicMock()
        glove_content = "word1 0.1 0.2 0.3\nword2 0.4 0.5 0.6\nword 0.9 0.9 0.9"
        self.open_mock = unittest.mock.mock_open(read_data=glove_content)
        with unittest.mock.patch('builtins.open', self.open_mock):
            self.query_embedding_vector = string_transforms.QueryEmbeddingVector(self.feature_info, self.file_io)

    def test_glove_file_loading(self):
        word1 = tf.constant("word1", dtype=tf.string)
        word2 = tf.constant("word2", dtype=tf.string)

        # Assert dimensions
        self.assertIn(str(word1), self.query_embedding_vector.word_vectors)
        self.assertIn(str(word2), self.query_embedding_vector.word_vectors)
        self.assertEqual(len(self.query_embedding_vector.word_vectors), 2)
        self.assertEqual(self.query_embedding_vector.embedding_dim, 3)

    def test_preprocess_text(self):
        text = tf.constant(["Hello, world!"])
        processed_text = self.query_embedding_vector.preprocess_text(text)
        self.assertEqual(processed_text.numpy().tolist(), [[b'hello', b'world']])

    def test_word_lookup_existing_word(self):
        word1 = tf.constant("word1", dtype=tf.string)
        embedding = self.query_embedding_vector.word_lookup(word1)
        expected_embedding = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        np.testing.assert_array_equal(embedding, expected_embedding)

    def test_word_lookup_unknown_word(self):
        embedding = self.query_embedding_vector.word_lookup("unknown_word")
        expected_embedding = np.zeros((3,), dtype=np.float32)
        np.testing.assert_array_equal(embedding, expected_embedding)

    def test_build_embeddings(self):
        query = tf.constant([["word1", "word2"]])
        query_embedding = self.query_embedding_vector.build_embeddings(query)
        expected_embedding = np.array([0.5, 0.7, 0.9], dtype=np.float32)
        np.isclose(query_embedding.numpy(), expected_embedding).all()