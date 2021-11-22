import unittest
import os
import shutil
import warnings
from ml4ir.base.data import ranklib_helper
import pandas as pd
import yaml
from ml4ir.base.features.feature_config import ExampleFeatureConfig, SequenceExampleFeatureConfig
from ml4ir.base.config.keys import TFRecordTypeKey
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.config.keys import DataFormatKey
from ml4ir.base.features.preprocessing import *
from ml4ir.base.io import file_io, local_io

warnings.filterwarnings("ignore")

INPUT_DIR = "ml4ir/applications/ranking/tests/data/ranklib/"
KEEP_ADDITIONAL_INFO = 0
NON_ZERO_FEATURES_ONLY = 0


class TestRanklibConversion(unittest.TestCase):
    def setUp(self):
        self.feature_config_yaml = INPUT_DIR + 'feature_config.yaml'
        self.feature_config_yaml_convert_to_clicks = INPUT_DIR + \
            'feature_config_convert_to_clicks.yaml'

    def tearDown(self):
        if os.path.exists(os.path.join(INPUT_DIR, "tfrecord")):
            shutil.rmtree(os.path.join(INPUT_DIR, "tfrecord"))

    def parse_config(self, tfrecord_type: str, feature_config, io) -> SequenceExampleFeatureConfig:
        if feature_config.endswith(".yaml"):
            feature_config = io.read_yaml(feature_config)
        else:
            feature_config = yaml.safe_load(feature_config)

        return SequenceExampleFeatureConfig(feature_config, None)

    def test_ranklib_in_ml4ir(self):
        """Creates a relevance dataset using ranklib format. Labels are graded relevance"""

        io = local_io.LocalIO()
        exFeatureConfig = self.parse_config(
            TFRecordTypeKey.SEQUENCE_EXAMPLE, self.feature_config_yaml, io)
        preprocessing_keys_to_fns = {}
        if 'preprocessing_info' in exFeatureConfig.get_label():
            if exFeatureConfig.get_label()['preprocessing_info'][0]['fn'] == 'convert_label_to_clicks':
                preprocessing_keys_to_fns['convert_label_to_clicks'] = convert_label_to_clicks

        dataset = RelevanceDataset(
            data_dir=INPUT_DIR,
            data_format=DataFormatKey.RANKLIB,
            feature_config=exFeatureConfig,
            tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
            batch_size=1,
            file_io=io,
            preprocessing_keys_to_fns=preprocessing_keys_to_fns,
            logger=None,
            keep_additional_info=KEEP_ADDITIONAL_INFO,
            non_zero_features_only=NON_ZERO_FEATURES_ONLY,
            max_sequence_size=319,
        )
        non_one_hot = False
        chk = [e for e in dataset.train]
        for e in chk:
            if sum(e[1][0]).numpy() > 1:
                non_one_hot = True
                break
        assert non_one_hot == True
        assert len(chk) == 49

        non_one_hot = False
        chk = [e for e in dataset.validation]
        for e in chk:
            if sum(e[1][0]).numpy() > 1:
                non_one_hot = True
                break
        assert non_one_hot == True
        assert len(chk) == 49

        non_one_hot = False
        chk = [e for e in dataset.test]
        for e in chk:
            if sum(e[1][0]).numpy() > 1:
                non_one_hot = True
                break
        assert non_one_hot == True
        assert len(chk) == 49

    def test_ranklib_in_ml4ir_click_conversion(self):
        """Creates a relevance dataset using ranklib format. Labels are converted to clicks graded relevance"""
        io = local_io.LocalIO()
        exFeatureConfig = self.parse_config(TFRecordTypeKey.SEQUENCE_EXAMPLE,
                                            self.feature_config_yaml_convert_to_clicks, io)
        preprocessing_keys_to_fns = {}
        if exFeatureConfig.get_label()['preprocessing_info'][0]['fn'] == 'convert_label_to_clicks':
            preprocessing_keys_to_fns['convert_label_to_clicks'] = convert_label_to_clicks

        dataset = RelevanceDataset(
            data_dir=INPUT_DIR,
            data_format=DataFormatKey.RANKLIB,
            feature_config=exFeatureConfig,
            tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
            batch_size=1,
            file_io=io,
            preprocessing_keys_to_fns=preprocessing_keys_to_fns,
            logger=None,
            keep_additional_info=KEEP_ADDITIONAL_INFO,
            non_zero_features_only=NON_ZERO_FEATURES_ONLY,
            max_sequence_size=319,
        )
        chk = [e for e in dataset.train]
        for e in chk:
            assert max(e[1][0]).numpy() == 1
        assert len(chk) == 49

        chk = [e for e in dataset.validation]
        for e in chk:
            assert max(e[1][0]).numpy() == 1
        assert len(chk) == 49

        chk = [e for e in dataset.test]
        for e in chk:
            assert max(e[1][0]).numpy() == 1
        assert len(chk) == 49


if __name__ == "__main__":
    unittest.main()
