import unittest
import tensorflow as tf
import logging

from ml4ir.base.data.tfrecord_reader import ExampleParser
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.features.preprocessing import PreprocessingMap

DATASET_PATH = "ml4ir/applications/classification/tests/data/tfrecord/train/file_0.tfrecord"
FEATURE_CONFIG_PATH = "ml4ir/applications/classification/tests/data/configs/feature_config.yaml"


class ExampleParserTest(unittest.TestCase):
    """
    Test class for ml4ir.base.data.tfrecord_reader.ExampleParser
    """

    def setUp(self):
        file_io = LocalIO()
        logger = logging.getLogger()

        self.dataset = tf.data.TFRecordDataset(DATASET_PATH)
        self.proto = next(iter(self.dataset))
        self.feature_config = FeatureConfig.get_instance(
            tfrecord_type=TFRecordTypeKey.EXAMPLE,
            feature_config_dict=file_io.read_yaml(FEATURE_CONFIG_PATH),
            logger=logger
        )
        self.se_parser = ExampleParser(feature_config=self.feature_config,
                                       preprocessing_map=PreprocessingMap(),
                                       required_fields_only=False)

    def test_features_spec(self):
        """
        Test the feature specification constructed and used to parse the Example proto
        """
        features_spec = self.se_parser.features_spec

        assert isinstance(features_spec, dict)

        # Check if the feature specification matches with the feature_config
        assert len(set(self.feature_config.get_all_features("name"))
                   ) == len(features_spec)
        for feature in self.feature_config.get_all_features("name"):
            assert feature in features_spec

    def test_extract_features_from_proto(self):
        """
        Test extraction of features from serialized proto
        """
        features = self.se_parser.extract_features_from_proto(
            self.proto)

        for feature, feature_tensor in features.items():
            # Test that no feature is a sparse tensor
            assert not isinstance(feature_tensor, tf.sparse.SparseTensor)

            # Test that each extracted feature is a scalar
            assert feature_tensor.shape == ()

        # Assert that there is no mask feature
        assert "mask" not in features

    def test_get_default_tensor(self):
        """
        Test the default tensor used for missing features
        """
        default_tensor = self.se_parser.get_default_tensor(
            self.feature_config.get_feature("query_text"))
        assert default_tensor.shape == ()

        default_tensor = self.se_parser.get_default_tensor(
            self.feature_config.get_feature("user_context"))
        assert default_tensor.shape == ()

    def test_get_feature(self):
        """
        Test fetching feature tensor from extracted feature dictionary
        """
        feature_tensor = self.se_parser.get_feature(self.feature_config.get_feature("query_text"),
                                                    {"query_text": tf.zeros((3, 4, 6))})
        assert feature_tensor.shape == (3, 4, 6)

        # Check missing feature being replaced with default tensor
        feature_tensor = self.se_parser.get_feature(self.feature_config.get_feature("query_text"),
                                                    {})
        assert feature_tensor.shape == ()

    def test_generate_and_add_mask(self):
        """
        Test mask generation and addition
        """
        features_dict, sequence_size = self.se_parser.generate_and_add_mask({}, {
        })

        assert "mask" not in features_dict
        assert sequence_size == tf.constant(0)

    def test_adjust_shape(self):
        """
        Test adjusting the shape of the feature tensor
        """
        feature_tensor = self.se_parser.adjust_shape(tf.zeros((9, 4, 2)),
                                                     self.feature_config.get_feature("query_text"))

        assert feature_tensor.shape == (1, 9, 4, 2)

    def pad_feature(self):
        """
        Test feature padding to max sequence size
        """
        feature_tensor = self.se_parser.pad_feature(tf.zeros((10)),
                                                    self.feature_config.get_feature("query_text"))

        # Check that there was no padding done
        assert feature_tensor.shape == (10,)
