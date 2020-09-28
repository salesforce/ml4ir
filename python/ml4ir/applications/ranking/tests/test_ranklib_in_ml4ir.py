import sys; sys.path.insert(0, '/Users/mohamed.m/Documents/work/projects/ml4ir/python/')
import unittest
import os
import warnings
from ml4ir.base.data import ranklib_to_ml4ir
import pandas as pd
import yaml
from ml4ir.base.features.feature_config import ExampleFeatureConfig
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.config.keys import DataFormatKey
from ml4ir.base.io import file_io, local_io

warnings.filterwarnings("ignore")

INPUT_DIR = "/Users/mohamed.m/Documents/work/projects/ml4ir/python/ml4ir/applications/ranking/tests/data/ranklib_test_data/"
QUERY_ID_NAME = 'qid'
RELEVANCE_NAME = 'relevance'
KEEP_ADDITIONAL_INFO = 0
GL_2_CLICKS = 1
NON_ZERO_FEATURES_ONLY = 0


class TestRanklibConversion(unittest.TestCase):
    def setUp(self):
        self.feature_config_yaml = INPUT_DIR + 'feature_config.yaml'

    def parse_config(self, tfrecord_type: str, feature_config, io) -> ExampleFeatureConfig:
        if feature_config.endswith(".yaml"):
            feature_config = io.read_yaml(feature_config)
        else:
            feature_config = yaml.safe_load(feature_config)

        return ExampleFeatureConfig(feature_config)

    def test_ranklib_in_ml4ir(self):
        io = local_io.LocalIO()
        exFeatureConfig = self.parse_config(TFRecordTypeKey.EXAMPLE, self.feature_config_yaml, io)
        dataset = RelevanceDataset(
            data_dir=INPUT_DIR,
            data_format=DataFormatKey.RANKLIB,
            feature_config=exFeatureConfig,
            tfrecord_type=TFRecordTypeKey.EXAMPLE,
            batch_size=32,
            file_io=io,
            preprocessing_keys_to_fns={},
            logger=None,
            keep_additional_info=KEEP_ADDITIONAL_INFO,
            gl_2_clicks=GL_2_CLICKS,
            non_zero_features_only=NON_ZERO_FEATURES_ONLY,
            max_sequence_size=319,
        )


        chk = [e for e in dataset.train]
        assert len(chk) == 156
        chk = [e for e in dataset.validation]
        assert len(chk) == 156
        chk = [e for e in dataset.test]
        assert len(chk) == 156




    def tearDown(self):
        pass


if __name__ == "__main__":
    unittest.main()
