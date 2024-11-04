import gc
import os
import pathlib
import unittest
import warnings

import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
from testfixtures import TempDirectory

from ml4ir.applications.ranking.config.parse_args import get_args
from ml4ir.applications.ranking.pipeline import RankingPipeline

warnings.filterwarnings("ignore")

ROOT_DATA_DIR = "ml4ir/applications/ranking/tests/data"


def train_ml4ir(data_dir, feature_config, model_config, logs_dir, aux_loss):
    argv = [
        "--data_dir",
        data_dir,
        "--feature_config",
        feature_config,
        "--loss_type",
        "listwise",
        "--scoring_type",
        "listwise",
        "--run_id",
        "test_aux_loss",
        "--data_format",
        "tfrecord",
        "--execution_mode",
        "train_evaluate",
        "--loss_key",
        "softmax_cross_entropy",
        "--aux_loss_key",
        aux_loss,
        "--aux_loss_weight",
        "0.2",
        "--num_epochs",
        "1",
        "--model_config",
        model_config,
        "--batch_size",
        "32",
        "--logs_dir",
        logs_dir,
        "--max_sequence_size",
        "25",
        "--train_pcent_split",
        "0.7",
        "--val_pcent_split",
        "0.15",
        "--test_pcent_split",
        "0.15",
        "--early_stopping_patience",
        "25",
        "--metrics_keys",
        "MRR",
        "categorical_accuracy",
        "--monitor_metric",
        "categorical_accuracy",
    ]
    args = get_args(argv)
    rp = RankingPipeline(args=args)
    rp.run()


class TestDualObjectiveTraining(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(__file__).parent
        self.working_dir = TempDirectory()
        self.log_dir = self.working_dir.makedir("logs")

    def tearDown(self):
        TempDirectory.cleanup_all()

        # Explicitly clear keras memory
        gc.collect()
        K.clear_session()

    def test_end_to_end_aux_one_hot_cross_entropy(self):
        feature_config_path = os.path.join(
            ROOT_DATA_DIR, "configs", "feature_config_aux_loss.yaml"
        )
        model_config_path = os.path.join(
            ROOT_DATA_DIR, "configs", "model_config_cyclic_lr.yaml")
        data_dir = os.path.join(ROOT_DATA_DIR, "tfrecord")
        aux_loss = "aux_one_hot_cross_entropy"
        train_ml4ir(data_dir, feature_config_path,
                    model_config_path, self.log_dir, aux_loss)

        results_dict = pd.read_csv(
            os.path.join(self.log_dir, "test_aux_loss", "_SUCCESS"), header=None
        ).set_index(0).to_dict()[1]

        assert np.isclose(float(results_dict["train_loss"]), 0.705109179019928, atol=0.0001)
        assert np.isclose(float(results_dict["train_primary_loss"]), 0.575706958770752, atol=0.0001)
        assert np.isclose(float(results_dict["train_aux_loss"]), 0.5712536573410034, atol=0.0001)

        assert np.isclose(float(results_dict["val_loss"]), 0.7087578773498535, atol=0.0001)
        assert np.isclose(float(results_dict["val_primary_loss"]), 0.4498528838157654, atol=0.0001)
        assert np.isclose(float(results_dict["val_aux_loss"]), 0.4466691017150879, atol=0.0001)

    def test_end_to_end_aux_softmax_cross_entropy(self):
        feature_config_path = os.path.join(
            ROOT_DATA_DIR, "configs", "feature_config_aux_loss.yaml"
        )
        model_config_path = os.path.join(
            ROOT_DATA_DIR, "configs", "model_config_cyclic_lr.yaml")
        data_dir = os.path.join(ROOT_DATA_DIR, "tfrecord")
        aux_loss = "aux_softmax_cross_entropy"
        train_ml4ir(data_dir, feature_config_path,
                    model_config_path, self.log_dir, aux_loss)

        results_dict = pd.read_csv(
            os.path.join(self.log_dir, "test_aux_loss", "_SUCCESS"), header=None
        ).set_index(0).to_dict()[1]

        assert np.isclose(float(results_dict["train_loss"]), 0.7075825929641724, atol=0.0001)
        assert np.isclose(float(results_dict["train_primary_loss"]), 0.57579505443573, atol=0.0001)
        assert np.isclose(float(results_dict["train_aux_loss"]), 0.5771192908287048, atol=0.0001)

        assert np.isclose(float(results_dict["val_loss"]), 0.7123104333877563, atol=0.0001)
        assert np.isclose(float(results_dict["val_primary_loss"]), 0.44926804304122925, atol=0.0001)
        assert np.isclose(float(results_dict["val_aux_loss"]), 0.45026063919067383, atol=0.0001)