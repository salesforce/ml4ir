import traceback
import unittest

import numpy as np
import pytest
import requests
from sentence_transformers import SentenceTransformer

from ml4ir.base.model.layers.sentence_transformers import SentenceTransformerWithTokenizerLayer


def connection_to_huggingface_failed():
    """Checks if python can connect to huggingface URL"""
    connection_failed = False
    try:
        requests.get("https://huggingface.co")
    except Exception as e:
        connection_failed = True
        traceback.print_exc()
    return connection_failed


class TestSentenceTransformerWithTokenizerLayer(unittest.TestCase):
    TEST_PHRASES = ["test query to test the embedding layer",
                    "Another test query which does more testing!"]

    @pytest.mark.skipif(connection_to_huggingface_failed(),
                        reason="Skipping because of error connecting to huggingface.co")
    def test_e5_base(self):
        model = SentenceTransformerWithTokenizerLayer(model_name_or_path="intfloat/e5-base")
        embeddings = model.encode(self.TEST_PHRASES)

        self.assertEqual(embeddings.shape, (2, 768))
        self.assertTrue(np.allclose(embeddings[0, :5], [-0.01958332, 0.02002536, 0.00893079, -0.02941261, 0.06580967]))
        self.assertTrue(np.allclose(embeddings[1, :5], [-0.0034735, 0.04219092, -0.00087385, -0.0156969, 0.06526384]))

    @pytest.mark.skipif(connection_to_huggingface_failed(),
                        reason="Skipping because of error connecting to huggingface.co")
    def test_distiluse(self):
        model = SentenceTransformerWithTokenizerLayer(
            model_name_or_path="sentence-transformers/distiluse-base-multilingual-cased-v1")
        embeddings = model.encode(self.TEST_PHRASES)

        self.assertEqual(embeddings.shape, (2, 512))
        print("debugging scores")
        print(embeddings[0, :5])
        self.assertTrue(np.allclose(embeddings[0, :5], [0.00174324, 0.01326917, -0.01836515, 0.05429127, 0.06062959]))
        self.assertTrue(
            np.allclose(embeddings[1, :5], [0.03018673, -0.00636012, -0.00495617, -0.04597681, -0.05619023]))

    @pytest.mark.skipif(connection_to_huggingface_failed(),
                        reason="Skipping because of error connecting to huggingface.co")
    def test_e5_base_with_sentence_transformers(self):
        model = SentenceTransformerWithTokenizerLayer(model_name_or_path="intfloat/e5-base")
        embeddings = model.encode(self.TEST_PHRASES)

        st_model = SentenceTransformer("intfloat/e5-base")
        st_embeddings = st_model.encode(self.TEST_PHRASES)

        self.assertTrue(np.allclose(embeddings, st_embeddings, atol=1e-5))

    @pytest.mark.skipif(connection_to_huggingface_failed(),
                        reason="Skipping because of error connecting to huggingface.co")
    def test_distiluse_with_sentence_transformers(self):
        model = SentenceTransformerWithTokenizerLayer(
            model_name_or_path="sentence-transformers/distiluse-base-multilingual-cased-v1")
        embeddings = model.encode(self.TEST_PHRASES)

        st_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")
        st_embeddings = st_model.encode(self.TEST_PHRASES)

        self.assertTrue(np.allclose(embeddings, st_embeddings, atol=1e-5))