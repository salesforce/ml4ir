import unittest
import tensorflow as tf
import logging

from ml4ir.base.data.tfrecord_reader import TFRecordSequenceExampleParser
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.features.preprocessing import PreprocessingMap

DATASET_PATH = "ml4ir/applications/ranking/tests/data/tfrecord/train/file_0.tfrecord"
FEATURE_CONFIG_PATH = "ml4ir/applications/ranking/tests/data/configs/feature_config.yaml"
MAX_SEQUENCE_SIZE = 25


class SequenceExampleParserTest(unittest.TestCase):
    """
    Test class for ml4ir.base.data.tfrecord_reader.TFRecordSequenceExampleParser
    """

    def setUp(self):
        file_io = LocalIO()
        logger = logging.getLogger()

        self.dataset = tf.data.TFRecordDataset(DATASET_PATH)
        self.proto = next(iter(self.dataset))
        self.feature_config = FeatureConfig.get_instance(
            tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
            feature_config_dict=file_io.read_yaml(FEATURE_CONFIG_PATH),
            logger=logger,
        )
        self.parser = TFRecordSequenceExampleParser(
            feature_config=self.feature_config,
            preprocessing_map=PreprocessingMap(),
            required_fields_only=False,
            pad_sequence=True,
            max_sequence_size=25,
        )

    def test_features_spec(self):
        """
        Test the feature specification constructed and used to parse the Example proto
        """
        features_spec = self.parser.features_spec

        assert isinstance(features_spec, tuple)
        assert isinstance(features_spec[0], dict)
        assert isinstance(features_spec[1], dict)

        # Check if the feature specification matches with the feature_config
        assert len(set(self.feature_config.get_context_features("name"))) == len(features_spec[0])
        assert len(set(self.feature_config.get_sequence_features("name"))) == len(features_spec[1])

        for feature in self.feature_config.get_context_features("name"):
            assert feature in features_spec[0]
        for feature in self.feature_config.get_sequence_features("name"):
            assert feature in features_spec[1]

    def test_extract_features_from_proto(self):
        """
        Test extraction of features from serialized proto
        """
        context_features, sequence_features = self.parser.extract_features_from_proto(self.proto)

        for feature in self.feature_config.get_context_features("name"):
            assert feature in context_features

            # Test that each extracted feature is a scalar
            assert context_features[feature].shape == ()

        for feature in self.feature_config.get_sequence_features("name"):
            assert feature in sequence_features

            # Test that all features are sparse tensor
            assert isinstance(sequence_features[feature], tf.sparse.SparseTensor)

            feature_tensor = tf.sparse.to_dense(tf.sparse.reset_shape(sequence_features[feature]))

            assert feature_tensor.shape == (1, 2)

        # Assert that there is no mask feature
        assert "mask" not in sequence_features

    def test_get_default_tensor(self):
        """
        Test the default tensor used for missing features
        """
        default_tensor = self.parser.get_default_tensor(
            self.feature_config.get_feature("query_text"), sequence_size=25
        )
        assert default_tensor.shape == ()

        default_tensor = self.parser.get_default_tensor(
            self.feature_config.get_feature("quality_score"), sequence_size=8
        )
        assert default_tensor.shape == (8,)

    def test_get_feature(self):
        """
        Test fetching feature tensor from extracted feature dictionary
        """
        # Checking context features
        feature_tensor = self.parser.get_feature(
            self.feature_config.get_feature("query_text"),
            extracted_features=({"query_text": tf.zeros((3, 4, 6))}, {}),
            sequence_size=10,
        )
        assert feature_tensor.shape == (1, 3, 4, 6)

        # Check missing feature being replaced with default tensor
        feature_tensor = self.parser.get_feature(
            self.feature_config.get_feature("query_text"),
            extracted_features=({}, {}),
            sequence_size=10,
        )
        assert feature_tensor.shape == (1,)

        # Checking sequence features
        feature_tensor = self.parser.get_feature(
            self.feature_config.get_feature("quality_score"),
            extracted_features=({}, {"quality_score": tf.zeros((3, 4, 6))}),
            sequence_size=10,
        )
        assert feature_tensor.shape == (3, 4, 6)

        # Check missing feature being replaced with default tensor
        feature_tensor = self.parser.get_feature(
            self.feature_config.get_feature("quality_score"),
            extracted_features=({}, {}),
            sequence_size=10,
        )
        assert feature_tensor.shape == (10,)

    def test_generate_and_add_mask(self):
        """
        Test mask generation and addition
        """
        rank_tensor = tf.constant([[1, 2, 3, 4, 5]])
        indices = tf.where(tf.not_equal(rank_tensor, tf.constant(0)))
        values = tf.gather_nd(rank_tensor, indices)
        sparse_rank_tensor = tf.SparseTensor(indices, values, rank_tensor.shape)

        # Check when pad sequence is set to True
        features_dict, sequence_size = self.parser.generate_and_add_mask(
            ({}, {"rank": sparse_rank_tensor}), {}
        )

        assert "mask" in features_dict
        assert features_dict["mask"].shape == (25,)
        assert tf.reduce_sum(features_dict["mask"]).numpy() == 5
        assert sequence_size == 25

        # Check when pad sequence is set to False
        self.parser.pad_sequence = False
        features_dict, sequence_size = self.parser.generate_and_add_mask(
            ({}, {"rank": sparse_rank_tensor}), {}
        )

        assert "mask" in features_dict
        assert features_dict["mask"].shape == (5,)
        assert tf.reduce_sum(features_dict["mask"]).numpy() == 5
        assert sequence_size == 5
        self.parser.pad_sequence = True

    def test_pad_feature(self):
        """
        Test feature padding to max sequence size
        """
        # Check no padding for context features
        feature_tensor = self.parser.pad_feature(tf.zeros((10)),
                                                 self.feature_config.get_feature("query_text"))
        assert feature_tensor.shape == (10,)

        # Check padding for sequence features
        feature_tensor = self.parser.pad_feature(tf.zeros((10)),
                                                 self.feature_config.get_feature("quality_score"))
        assert feature_tensor.shape == (25,)

    def test_parse_fn(self):
        """
        Test the Example parsing function
        """
        # Check tensor shapes when pad_sequence is True
        features, labels = self.parser.get_parse_fn()(self.proto)

        assert isinstance(features, dict)
        assert isinstance(labels, tf.Tensor)

        for feature in self.feature_config.get_all_features(key="node_name", include_label=False):
            assert feature in features

        assert features["mask"].shape == (25,)
        for feature in self.feature_config.get_context_features("node_name"):
            assert features[feature].shape == (1,)
        for feature in self.feature_config.get_sequence_features("node_name"):
            if feature != "clicked":
                assert features[feature].shape == (25,)
        assert labels.shape == (25,)

        # Check tensor shapes when pad_sequence is False
        self.parser.pad_sequence = False
        features, labels = self.parser.get_parse_fn()(self.proto)

        assert features["mask"].shape == (2,)
        for feature in self.feature_config.get_context_features("node_name"):
            assert features[feature].shape == (1,)
        for feature in self.feature_config.get_sequence_features("node_name"):
            if feature != "clicked":
                assert features[feature].shape == (2,)
        assert labels.shape == (2,)
        self.pad_sequence = True
