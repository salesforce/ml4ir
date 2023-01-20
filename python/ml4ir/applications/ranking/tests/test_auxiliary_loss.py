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
        "--primary_loss_weight",
        "0.8",
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
        "RankMatchFailure",
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

    def test_E2E_aux_softmax_CE(self):
        feature_config_path = os.path.join(
            ROOT_DATA_DIR, "configs", "feature_config_aux_loss.yaml"
        )
        model_config_path = os.path.join(ROOT_DATA_DIR, "configs", "model_config_cyclic_lr.yaml")
        data_dir = os.path.join(ROOT_DATA_DIR, "tfrecord")
        aux_loss = "softmax_cross_entropy"
        train_ml4ir(data_dir, feature_config_path, model_config_path, self.log_dir, aux_loss)

        ml4ir_results = pd.read_csv(
            os.path.join(self.log_dir, "test_aux_loss", "_SUCCESS"), header=None
        )
        primary_training_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "train_ranking_score_loss"][1]
        )
        assert np.isclose(primary_training_loss, 1.1877643, atol=0.0001)
        aux_training_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "train_aux_ranking_score_loss"][1]
        )
        assert np.isclose(aux_training_loss, 1.2242277, atol=0.0001)
        primary_val_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "val_ranking_score_loss"][1]
        )
        assert np.isclose(primary_val_loss, 1.2086908, atol=0.0001)
        aux_val_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "val_aux_ranking_score_loss"][1]
        )
        # RankMatchFailure metric comparisons
        assert np.isclose(aux_val_loss, 1.2806674, atol=0.0001)
        # RankMatchFailure metric comparisons
        aux_val_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "train_ranking_score_old_RankMatchFailure"][1]
        )
        assert np.isclose(aux_val_loss, 0.00082716695, atol=0.00001)
        aux_val_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "train_ranking_score_new_RankMatchFailure"][1]
        )
        assert np.isclose(aux_val_loss, 0.00012763393, atol=0.00001)
        aux_val_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "val_ranking_score_old_RankMatchFailure"][1]
        )
        assert np.isclose(aux_val_loss, 0.0011314502, atol=0.00001)
        aux_val_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "val_ranking_score_new_RankMatchFailure"][1]
        )
        assert np.isclose(aux_val_loss, 0.0002121307, atol=0.00001)

    def test_E2E_aux_basic_CE(self):
        feature_config_path = os.path.join(
            ROOT_DATA_DIR, "configs", "feature_config_aux_loss.yaml"
        )
        model_config_path = os.path.join(ROOT_DATA_DIR, "configs", "model_config_cyclic_lr.yaml")
        data_dir = os.path.join(ROOT_DATA_DIR, "tfrecord")
        aux_loss = "basic_cross_entropy"
        train_ml4ir(data_dir, feature_config_path, model_config_path, self.log_dir, aux_loss)

        ml4ir_results = pd.read_csv(
            os.path.join(self.log_dir, "test_aux_loss", "_SUCCESS"), header=None
        )
        primary_training_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "train_ranking_score_loss"][1]
        )
        assert np.isclose(primary_training_loss, 1.1911143, atol=0.0001)
        aux_training_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "train_aux_ranking_score_loss"][1]
        )
        assert np.isclose(aux_training_loss, 0.3824733, atol=0.0001)
        primary_val_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "val_ranking_score_loss"][1]
        )
        assert np.isclose(primary_val_loss, 1.2130133, atol=0.0001)
        aux_val_loss = float(
            ml4ir_results.loc[ml4ir_results[0] == "val_aux_ranking_score_loss"][1]
        )
        assert np.isclose(aux_val_loss, 0.3906489, atol=0.0001)


if __name__ == "__main__":
    unittest.main()
