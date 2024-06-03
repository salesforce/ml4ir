import unittest
from unittest.mock import MagicMock
import tensorflow as tf
import pathlib
import yaml
from ml4ir.base.model.scoring.monte_carlo_scorer import MonteCarloScorer


class TestMonteCarloScorer(unittest.TestCase):

    def setUp(self):
        model_config_file = pathlib.Path(__file__).parent / "data" / "configs" / "model_config_monte_carlo_10.yaml"

        with open(model_config_file.as_posix(), 'r') as f:
            model_config = yaml.safe_load(f)

        self.scorer = MonteCarloScorer(
            model_config=model_config,
            feature_config=MagicMock(),
            interaction_model=MagicMock(),
            loss=MagicMock(),
            file_io=MagicMock()
        )

        self.scorer.architecture_op = MagicMock()

    def test_10_calls(self):
        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = MonteCarloScorer.call(self.scorer, inputs, training=None)

        # Check that the call method has looped the correct number of times
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_inference_trials)

    def test_0_calls(self):
        self.scorer.monte_carlo_inference_trials = 0

        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = MonteCarloScorer.call(self.scorer, inputs, training=None)

        # Check that the call method has looped the correct number of times
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_inference_trials)


if __name__ == "__main__":
    unittest.main()