import unittest
import tensorflow as tf
import logging

from ml4ir.base.data.tfrecord_reader import TFRecordExampleParser
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.features.preprocessing import PreprocessingMap

DATASET_PATH = "ml4ir/applications/classification/tests/data/tfrecord/train/file_0.tfrecord"
FEATURE_CONFIG_PATH = "ml4ir/applications/classification/tests/data/configs/feature_config.yaml"


class TFRecordExampleParserTest(unittest.TestCase):
    """
    Test class for ml4ir.base.data.tfrecord_reader.TFRecordExampleParser
    """

    def setUp(self):
        file_io = LocalIO()
        logger = logging.getLogger()

        self.dataset = tf.data.TFRecordDataset(DATASET_PATH)
        self.proto = next(iter(self.dataset))
        self.feature_config = FeatureConfig.get_instance(
            tfrecord_type=TFRecordTypeKey.EXAMPLE,
            feature_config_dict=file_io.read_yaml(FEATURE_CONFIG_PATH),
            logger=logger,
        )
        self.parser = TFRecordExampleParser(
            feature_config=self.feature_config,
            preprocessing_map=PreprocessingMap(),
            required_fields_only=False,
        )

    def test_features_spec(self):
        """
        Test the feature specification constructed and used to parse the Example proto
        """
        features_spec = self.parser.features_spec

        assert isinstance(features_spec, dict)

        # Check if the feature specification matches with the feature_config
        assert len(set(self.feature_config.get_all_features("name"))) == len(features_spec)
        for feature in self.feature_config.get_all_features("name"):
            assert feature in features_spec

    def test_extract_features_from_proto(self):
        """
        Test extraction of features from serialized proto
        """
        features = self.parser.extract_features_from_proto(self.proto)

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
        default_tensor = self.parser.get_default_tensor(
            self.feature_config.get_feature("query_text")
        )
        assert default_tensor.shape == ()

        default_tensor = self.parser.get_default_tensor(
            self.feature_config.get_feature("user_context")
        )
        assert default_tensor.shape == ()

    def test_get_feature(self):
        """
        Test fetching feature tensor from extracted feature dictionary
        """
        feature_tensor = self.parser.get_feature(
            self.feature_config.get_feature("query_text"), {"query_text": tf.zeros((3, 4, 6))}
        )
        assert feature_tensor.shape == (1, 3, 4, 6)

        # Check missing feature being replaced with default tensor
        feature_tensor = self.parser.get_feature(self.feature_config.get_feature("query_text"), {})
        assert feature_tensor.shape == (1,)

    def test_generate_and_add_mask(self):
        """
        Test mask generation and addition
        """
        features_dict, sequence_size = self.parser.generate_and_add_mask({}, {})

        assert "mask" not in features_dict
        assert sequence_size == tf.constant(0)

    def test_pad_feature(self): 
        """ 
        Test feature padding to max sequence size   
        """ 
        feature_tensor = self.parser.pad_feature(tf.zeros((10)),    
                                                 self.feature_config.get_feature("query_text")) 

        # Check that there was no padding done  
        assert feature_tensor.shape == (10,)

    def test_parse_fn(self):
        """
        Test the Example parsing function
        """
        features, labels = self.parser.get_parse_fn()(self.proto)

        assert isinstance(features, dict)
        assert isinstance(labels, tf.Tensor)

        for feature in self.feature_config.get_all_features(key="node_name", include_label=False):
            assert feature in features

        # Check tensor shapes
        assert features["query_key"].shape == (1,)
        assert features["query_text"].shape == (1,)
        assert features["query_words"].shape == (1, 20)
        assert features["domain_id"].shape == (1,)
        assert features["user_context"].shape == (1, 20)
        assert labels.shape == (1,)
