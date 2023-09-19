import tensorflow as tf
import numpy as np
import copy

from ml4ir.applications.ranking.features.feature_fns import categorical
from ml4ir.applications.ranking.features.feature_fns import normalization
from ml4ir.base.tests.test_base import RelevanceTestBase


FEATURE_INFO = {
    "name": "test_feature",
    "feature_layer_info": {
        "args": {
            "vocabulary_file": "ml4ir/applications/ranking/tests/data/configs/domain_name_vocab_no_id.csv",
            "num_oov_buckets": 1
        },
    },
    "default_value": "",
}


class FeatureLayerTest(RelevanceTestBase):

    def test_sequence_categorical_vector_embedding(self):
        """
        Asserts the conversion of a categorical string tensor into a categorical embedding
        Works by converting the string into indices using a vocabulary file and then
        converting the indices into categorical embeddings

        The embedding dimensions, buckets, etc are controlled by the feature_info
        """
        embedding_weights = np.random.randn(5 + 1, 32)
        feature_info = copy.deepcopy(FEATURE_INFO)
        feature_info["feature_layer_info"]["args"]["output_mode"] = "embedding"
        feature_info["feature_layer_info"]["args"]["embedding_size"] = 32
        feature_info["feature_layer_info"]["args"]["embeddings_initializer"] = tf.keras.initializers.Constant(embedding_weights)

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        actual_embedding = categorical.SequenceCategoricalVector(
            feature_info, self.file_io
        )(string_tensor).numpy()

        # Check if the embedding vectors match what we expect from the preset embedding weights
        # NOTE - Out of vocabulary tokens are mapped to 0 index
        domain_0_embedding = embedding_weights[1]
        domain_1_embedding = embedding_weights[2]
        domain_2_embedding = embedding_weights[3]
        oov_embedding = embedding_weights[0]
        expected_embedding = np.stack([domain_0_embedding,
                                       domain_1_embedding,
                                       domain_0_embedding,
                                       domain_2_embedding,
                                       oov_embedding,
                                       oov_embedding])
        self.assertTrue(np.isclose(actual_embedding, expected_embedding).all())

    def test_sequence_categorical_vector_one_hot(self):
        """
        Asserts the conversion of a categorical string tensor into a one-hot representation
        Works by converting the string into indices using a vocabulary file and then
        converting the indices into one-hot vectors

        The one-hot vector dimensions, buckets, etc are controlled by the feature_info
        """
        feature_info = copy.deepcopy(FEATURE_INFO)
        feature_info["feature_layer_info"]["args"]["output_mode"] = "one_hot"

        # Define an input string tensor
        string_tensor = tf.constant(
            ["domain_0", "domain_1", "domain_0", "domain_2", "domain_10", "domain_11"]
        )

        actual_one_hot = categorical.SequenceCategoricalVector(
            feature_info, self.file_io
        )(string_tensor).numpy()

        # Check if the one hot vectors match what we expect
        # NOTE - Out of vocabulary tokens are mapped to 0 index
        domain_0_one_hot = np.array([0., 1., 0., 0., 0., 0.])
        domain_1_one_hot = np.array([0., 0., 1., 0., 0., 0.])
        domain_2_one_hot = np.array([0., 0., 0., 1., 0., 0.])
        oov_one_hot = np.array([1., 0., 0., 0., 0., 0.])
        expected_one_hot = np.stack([domain_0_one_hot,
                                     domain_1_one_hot,
                                     domain_0_one_hot,
                                     domain_2_one_hot,
                                     oov_one_hot,
                                     oov_one_hot])
        self.assertTrue(np.isclose(actual_one_hot, expected_one_hot).all())

    def test_theoretical_min_max_norm(self):
        """
        Test TheoreticalMinMaxNormalization feature transform
        """
        feature_info = copy.deepcopy(FEATURE_INFO)
        feature_info["feature_layer_info"]["args"]["theoretical_min"] = 0.5

        input_feature = np.array([[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

        actual_normed_feature = normalization.TheoreticalMinMaxNormalization(
            feature_info, self.file_io
        )(input_feature).numpy()

        expected_normed_feature = np.array([[0., 0., 0., 0., 0., 0., 0., 0., 0.2, 0.4, 0.6, 0.8, 1.]])[:, :, np.newaxis]

        self.assertTrue(np.isclose(actual_normed_feature, expected_normed_feature).all())

    def test_theoretical_min_max_norm_default_min(self):
        """
        Test TheoreticalMinMaxNormalization feature transform
        """
        feature_info = copy.deepcopy(FEATURE_INFO)

        input_feature = np.array([[-0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]])

        actual_normed_feature = normalization.TheoreticalMinMaxNormalization(
            feature_info, self.file_io
        )(input_feature).numpy()

        expected_normed_feature = np.array([[0., 0., 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]]).reshape(1, -1, 1)

        self.assertTrue(np.isclose(actual_normed_feature, expected_normed_feature).all())
