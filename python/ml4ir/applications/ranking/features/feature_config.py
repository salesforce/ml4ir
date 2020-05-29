import yaml
from logging import Logger

from ml4ir.base.features.feature_config import SequenceExampleFeatureConfig
import ml4ir.base.io.file_io as file_io

from typing import Optional


class RankingFeatureConfig(SequenceExampleFeatureConfig):
    def extract_features(self, features_dict, logger: Optional[Logger] = None):
        try:
            self.rank = features_dict.get("rank")
            self.all_features.append(self.rank)
        except KeyError:
            self.rank = None
            if logger:
                logger.warning("'rank' key not found in the feature_config specified")


def parse_config(feature_config, logger: Optional[Logger] = None) -> RankingFeatureConfig:
    if feature_config.endswith(".yaml"):
        feature_config = file_io.read_yaml(feature_config)
        if logger:
            logger.info("Reading feature config from YAML file : {}".format(feature_config))
    else:
        feature_config = yaml.safe_load(feature_config)
        if logger:
            logger.info("Reading feature config from YAML string : {}".format(feature_config))

    return SequenceExampleFeatureConfig(feature_config, logger=logger)
