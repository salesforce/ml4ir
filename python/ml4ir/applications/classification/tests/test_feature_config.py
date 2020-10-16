import unittest
import logging
import yaml
from ml4ir.base.features.feature_config import FeatureConfig, ExampleFeatureConfig
from ml4ir.base.config.keys import TFRecordTypeKey


FEAT_CONFIG = "ml4ir/applications/classification/tests/data/configs/feature_config.yaml"


class FeatureConfigTest(unittest.TestCase):
    """Test for Feature Config"""
    def setUp(self,):
        logger = logging.getLogger()
        self.feature_config = FeatureConfig.get_instance(
            tfrecord_type=TFRecordTypeKey.EXAMPLE,
            feature_config_dict=yaml.safe_load(open(FEAT_CONFIG)),
            logger=logger)

    def test_get_all_values(self):
        vocabulary_files = self.feature_config.get_all_values('vocabulary_file')
        assert vocabulary_files == ['ml4ir/applications/classification/tests/data/configs/vocabulary/entity_id.csv',
                                    'ml4ir/applications/classification/tests/data/configs/vocabulary/query_word.csv',
                                    'ml4ir/applications/classification/tests/data/configs/vocabulary/domain_id.csv',
                                    'ml4ir/applications/classification/tests/data/configs/vocabulary/entity_id.csv']

        vocabulary_files = self.feature_config.get_all_values('vocabulary_file_na')
        self.assertFalse(vocabulary_files)  # Empty dicts/lists assert to False

    def test_get_feature(self):
        feature = self.feature_config.get_feature('query_text')
        assert feature == {'default_value': '',
                           'dtype': 'string',
                           'feature_layer_info': {'args': {'embedding_size': 128,
                                                           'encoding_size': 128,
                                                           'encoding_type': 'bilstm',
                                                           'max_length': 20},
                                                  'fn': 'bytes_sequence_to_encoding_bilstm',
                                                  'shape': None,
                                                  'type': 'numeric'},
                           'log_at_inference': True,
                           'name': 'query_text',
                           'node_name': 'query_text',
                           'preprocessing_info': [{'args': {'remove_punctuation': True, 'to_lower': True},
                                                   'fn': 'preprocess_text'}],
                           'serving_info': {'name': 'query_text', 'required': True},
                           'trainable': True}

    def test_get_missing_feature(self):
        self.assertRaises(KeyError,
                          lambda: self.feature_config.get_feature('query_text_missing'))