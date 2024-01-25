import unittest

import numpy as np
from sentence_transformers import SentenceTransformer

from ml4ir.base.model.layers.sentence_transformers import SentenceTransformerWithTokenizerLayer


class TestSentenceTransformerWithTokenizerLayer(unittest.TestCase):
    TEST_PHRASES = ["test query to test the embedding layer",
                    "Another test query which does more testing!"]

    def test_e5_base(self):
        model = SentenceTransformerWithTokenizerLayer(model_name_or_path="intfloat/e5-base")
        embeddings = model(self.TEST_PHRASES).numpy()

        self.assertEqual(embeddings.shape, (2, 768))
        self.assertTrue(np.allclose(embeddings[0, :5], [-0.01958332, 0.02002536, 0.00893079, -0.02941261, 0.06580967]))
        self.assertTrue(np.allclose(embeddings[1, :5], [-0.0034735, 0.04219092, -0.00087385, -0.0156969, 0.06526384]))

    def test_distiluse(self):
        model = SentenceTransformerWithTokenizerLayer(
            model_name_or_path="sentence-transformers/distiluse-base-multilingual-cased-v1")
        embeddings = model(self.TEST_PHRASES).numpy()

        self.assertEqual(embeddings.shape, (2, 512))
        self.assertTrue(np.allclose(embeddings[0, :5], [0.00174321, 0.01326918, -0.01836516, 0.05429131, 0.06062959]))
        self.assertTrue(
            np.allclose(embeddings[1, :5], [0.03018673, -0.00636012, -0.00495617, -0.04597681, -0.05619023]))

    def test_e5_base_with_sentence_transformers(self):
        model = SentenceTransformerWithTokenizerLayer(model_name_or_path="intfloat/e5-base")
        embeddings = model(self.TEST_PHRASES).numpy()

        st_model = SentenceTransformer("intfloat/e5-base")
        st_embeddings = st_model.encode(self.TEST_PHRASES)

        self.assertTrue(np.allclose(embeddings, st_embeddings, atol=1e-5))

    def test_distiluse_with_sentence_transformers(self):
        model = SentenceTransformerWithTokenizerLayer(
            model_name_or_path="sentence-transformers/distiluse-base-multilingual-cased-v1")
        embeddings = model(self.TEST_PHRASES).numpy()

        st_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        st_embeddings = st_model.encode(self.TEST_PHRASES)

        self.assertTrue(np.allclose(embeddings, st_embeddings, atol=1e-5))

    def test_trainable(self):
        model = SentenceTransformerWithTokenizerLayer(
            model_name_or_path="sentence-transformers/distiluse-base-multilingual-cased-v1",
            trainable=True)
        model(self.TEST_PHRASES)
        self.assertTrue(model.trainable)
        self.assertTrue(model.transformer_model.trainable)
        self.assertTrue(model.dense.trainable)
        self.assertTrue(len(model.transformer_model.trainable_weights) > 0)
        self.assertTrue(len(model.dense.trainable_weights) > 0)

        model = SentenceTransformerWithTokenizerLayer(
            model_name_or_path="sentence-transformers/distiluse-base-multilingual-cased-v1",
            trainable=False)
        model(self.TEST_PHRASES)
        self.assertFalse(model.trainable)
        self.assertFalse(model.transformer_model.trainable)
        self.assertFalse(model.dense.trainable)
        self.assertTrue(len(model.transformer_model.trainable_weights) == 0)
        self.assertTrue(len(model.dense.trainable_weights) == 0)
