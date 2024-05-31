import unittest
import warnings
import pandas as pd
import numpy as np
import pathlib
from testfixtures import TempDirectory
import gc

import tensorflow.keras.backend as K

from ml4ir.applications.ranking.pipeline import RankingPipeline
from ml4ir.applications.ranking.config.parse_args import get_args

warnings.filterwarnings("ignore")


def ml4ir_sanity_pipeline(df, working_dir, log_dir, model_config):
    df.to_csv(working_dir / 'train' / 'data.csv')
    df.to_csv(working_dir / 'validation' / 'data.csv')
    df.to_csv(working_dir / 'test' / 'data.csv')

    fconfig_name = "feature_config_monte_carlo.yaml"
    feature_config_file = pathlib.Path(__file__).parent / "data" / "configs" / fconfig_name
    model_config_file = pathlib.Path(__file__).parent / "data" / "configs" / model_config
    eval_config_file = pathlib.Path(__file__).parent / "data" / "configs" / "evaluation_config.yaml"

    train_test_MC_ml4ir(working_dir.as_posix(), feature_config_file.as_posix(), model_config_file.as_posix(),
                        log_dir.as_posix(), eval_config_file.as_posix())

    ml4ir_results = pd.read_csv(log_dir / 'test_command_line'/ '_SUCCESS', header=None)
    ml4ir_mrr = float(ml4ir_results.loc[ml4ir_results[0] == 'test_new_MRR'][1])
    return ml4ir_mrr


def train_test_MC_ml4ir(data_dir, feature_config, model_config, logs_dir, eval_config_file):
    argv = ["--data_dir", data_dir,
            "--feature_config", feature_config,
            "--evaluation_config", eval_config_file,
            "--loss_key", "aux_one_hot_cross_entropy",
            "--run_id", "test_command_line",
            "--data_format", "csv",
            "--execution_mode", "train_evaluate",
            "--num_epochs", "1",
            "--model_config", model_config,
            "--batch_size", "1",
            "--logs_dir", logs_dir,
            "--max_sequence_size", "25",
            "--train_pcent_split", "0.7",
            "--val_pcent_split", "0.15",
            "--test_pcent_split", "0.15",
            "--early_stopping_patience", "25",
            "--metrics_keys", "MRR",
            "--monitor_metric", "MRR"]
    args = get_args(argv)
    rp = RankingPipeline(args=args)
    rp.run()


def run_MC_test(model_config_name, fname, working_dir, log_dir):
    df = pd.read_csv(pathlib.Path(__file__).parent / "data" / "linear_sanity_tests" / fname)
    ml4ir_mrr = ml4ir_sanity_pipeline(df, working_dir, log_dir, model_config_name)
    return ml4ir_mrr


class TestML4IRSanity(unittest.TestCase):
    def setUp(self):
        self.dir = pathlib.Path(__file__).parent
        self.working_dir = TempDirectory()
        self.log_dir = self.working_dir.makedir('logs')
        self.working_dir.makedir('train')
        self.working_dir.makedir('test')
        self.working_dir.makedir('validation')

    def tearDown(self):
        TempDirectory.cleanup_all()

        # Explicitly clear keras memory
        gc.collect()
        K.clear_session()

    def test_linear_ml4ir_sanity_1(self):
        # Test MC trials = 10
        ml4ir_mrr = run_MC_test(model_config_name="model_config_monte_carlo_10.yaml", fname="dataset_small.csv",
                        working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))

        assert np.isclose(ml4ir_mrr, 0.9545454545454546, atol=0.001)

    def test_linear_ml4ir_sanity_2(self):
        # Test No MC
        ml4ir_mrr = run_MC_test(model_config_name="model_config_monte_carlo_0.yaml", fname="dataset_small.csv",
                    working_dir=pathlib.Path(self.working_dir.path), log_dir=pathlib.Path(self.log_dir))
        assert np.isclose(ml4ir_mrr, 0.521509209744504, atol=0.001)


if __name__ == "__main__":
    unittest.main()
