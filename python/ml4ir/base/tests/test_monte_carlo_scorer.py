import unittest
from unittest.mock import MagicMock
import tensorflow as tf
import pathlib
import yaml
from ml4ir.base.model.scoring.monte_carlo_scorer import MonteCarloScorer


class TestMonteCarloScorer(unittest.TestCase):

    def setUp(self):
        model_config = {'architecture_key': 'linear',
                        'layers': [{'type': 'dense', 'name': 'linear_layer', 'units': 1, 'activation': None}],
                        'optimizer': {'key': 'adam'}, 'lr_schedule': {'key': 'constant', 'learning_rate': 0.01},
                        'monte_carlo_inference_trials': {'num_test_trials': 10, "num_training_trials": 5}}

        self.scorer = MonteCarloScorer(
            model_config=model_config,
            feature_config=MagicMock(),
            interaction_model=MagicMock(),
            loss=MagicMock(),
            file_io=MagicMock()
        )

        self.scorer.architecture_op = MagicMock()

    def test_10_calls_for_testing(self):
        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = self.scorer.call(self.scorer, inputs)

        # Check that the call method has looped the correct number of times.
        # It checks the number of "addition" operation used in aggregating the MC trials.
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_test_trials)

    def test_0_calls_for_testing(self):
        self.scorer.monte_carlo_test_trials = 0

        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = self.scorer.call(self.scorer, inputs)

        # Check that the call method has looped the correct number of times.
        # It checks the number of "addition" operation used in aggregating the MC trials.
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_test_trials)

    def test_10_calls_for_training(self):
        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = self.scorer.call(self.scorer, inputs)

        # Check that the call method has looped the correct number of times.
        # It checks the number of "addition" operation used in aggregating the MC trials.
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_training_trials)

    def test_0_calls_for_training(self):
        self.scorer.monte_carlo_training_trials = 0

        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = self.scorer.call(self.scorer, inputs)

        # Check that the call method has looped the correct number of times.
        # It checks the number of "addition" operation used in aggregating the MC trials.
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_training_trials)

    # TODO add test to check the actual scores


if __name__ == "__main__":
    unittest.main()
