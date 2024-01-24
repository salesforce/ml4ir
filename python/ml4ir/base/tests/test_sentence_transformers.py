import unittest

import numpy as np

from ml4ir.base.model.layers.sentence_transformers import SentenceTransformerWithTokenizerLayer


class TestSentenceTransformerWithTokenizerLayer(unittest.TestCase):

    def test_default_setting(self):
        model = SentenceTransformerWithTokenizerLayer()
        embeddings = model(["test query to test the embedding layer",
                            "Another test query which does more testing!"]).numpy()

        self.assertEqual(embeddings.shape, (2, 768))
        self.assertTrue(np.allclose(embeddings[0, :5], [-0.25825006, 0.26407936, 0.11777245, -0.38787124, 0.8678482]))
        self.assertTrue(np.allclose(embeddings[1, :5], [-0.04476589, 0.54374915, -0.01126207, -0.20229892, 0.8411089 ]))

    def test_normalize_embeddings(self):
        model = SentenceTransformerWithTokenizerLayer(normalize_embeddings=True)
        embeddings = model(["test query to test the embedding layer",
                            "Another test query which does more testing!"]).numpy()

        self.assertEqual(embeddings.shape, (2, 768))
        self.assertTrue(np.allclose(embeddings[0, :5], [-0.01958332, 0.02002536, 0.00893079, -0.02941261, 0.06580967]))
        self.assertTrue(np.allclose(embeddings[1, :5], [-0.0034735 , 0.04219092, -0.00087385, -0.0156969 , 0.06526384]))

    def test_trainable(self):
        model = SentenceTransformerWithTokenizerLayer(trainable=True)
        self.assertTrue(model.trainable)
        self.assertTrue(model.sentence_transformer.trainable)
        self.assertTrue(len(model.sentence_transformer.trainable_weights) > 0)

        model = SentenceTransformerWithTokenizerLayer(trainable=False)
        self.assertFalse(model.trainable)
        self.assertFalse(model.sentence_transformer.trainable)
        self.assertTrue(len(model.sentence_transformer.trainable_weights) == 0)
