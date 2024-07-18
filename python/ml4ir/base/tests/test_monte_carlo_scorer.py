import unittest
from unittest.mock import MagicMock
import tensorflow as tf
from ml4ir.base.model.scoring.monte_carlo_scorer import MonteCarloScorer


class TestMonteCarloScorer(unittest.TestCase):

    def setUp(self):
        self.model_config = {'architecture_key': 'linear',
                        'layers': [{'type': 'dense', 'name': 'linear_layer', 'units': 1, 'activation': None}],
                        'optimizer': {'key': 'adam'}, 'lr_schedule': {'key': 'constant', 'learning_rate': 0.01},
                        'monte_carlo_trials': {'num_test_trials': 10, "num_training_trials": 5}}

        self.scorer = MonteCarloScorer(
            model_config=self.model_config,
            feature_config=MagicMock(),
            interaction_model=MagicMock(),
            loss=MagicMock(),
            file_io=MagicMock()
        )

        self.scorer.architecture_op = MagicMock()

    def test_10_calls_for_testing(self):
        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = self.scorer.call(inputs, training=False)

        # Check that the call method has looped the correct number of times.
        # It checks the number of "addition" operation used in aggregating the MC trials.
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_test_trials)

    def test_0_calls_for_testing(self):
        self.scorer.monte_carlo_test_trials = 0

        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = self.scorer.call(inputs, training=False)

        # Check that the call method has looped the correct number of times.
        # It checks the number of "addition" operation used in aggregating the MC trials.
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_test_trials)

    def test_5_calls_for_training(self):
        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = self.scorer.call(inputs, training=True)

        # Check that the call method has looped the correct number of times.
        # It checks the number of "addition" operation used in aggregating the MC trials.
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_training_trials)

    def test_0_calls_for_training(self):
        self.scorer.monte_carlo_training_trials = 0

        inputs = {"f0": tf.constant([0.5], dtype=tf.float32)}

        # Call the method
        result = self.scorer.call(inputs, training=True)

        # Check that the call method has looped the correct number of times.
        # It checks the number of "addition" operation used in aggregating the MC trials.
        self.assertEqual(str(result["score"].name).count("iadd"), self.scorer.monte_carlo_training_trials)

    def test_fixed_mask_inputs(self):
        self.scorer = MonteCarloScorer(
            model_config=self.model_config,
            feature_config=MagicMock(),
            interaction_model=MagicMock(),
            loss=MagicMock(),
            file_io=MagicMock()
        )

        self.scorer.architecture_op = MagicMock()
        self.scorer.features_with_fixed_masks = ['feature1', 'feature2']
        self.scorer.fixed_mask = [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]]
        ]
        self.scorer.fixed_feature_count = 2

        inputs = {
            'feature1': tf.constant([[1.0, 2.0], [3.0, 4.0]]),
            'feature2': tf.constant([[5.0, 6.0], [7.0, 8.0]])
        }
        mask_count = 2

        expected_masked_inputs_list = [
            {
                'feature1': tf.constant([[1.0, 0.0], [3.0, 0.0]]),
                'feature2': tf.constant([[0.0, 6.0], [0.0, 8.0]])
            },
            {
                'feature1': tf.constant([[0.0, 2.0], [0.0, 4.0]]),
                'feature2': tf.constant([[5.0, 0.0], [7.0, 0.0]])
            }
        ]
        result = self.scorer.mask_inputs(inputs, mask_count)

        for i in range(mask_count):
            for key in inputs.keys():
                self.assertTrue(tf.reduce_all(tf.equal(result[i][key], expected_masked_inputs_list[i][key])))

    def test_reshape_and_normalize(self):
        self.scorer = MonteCarloScorer(
            model_config=self.model_config,
            feature_config=MagicMock(),
            interaction_model=MagicMock(),
            loss=MagicMock(),
            file_io=MagicMock()
        )

        self.scorer.architecture_op = MagicMock()
        self.scorer.monte_carlo_fixed_trials_count = 2
        mask_count = 2
        batch_size = 3

        # Create a mock all_scores tensor with shape [mask_count * batch_size, feature_count]
        all_scores = tf.constant([
            [1.0, 2.0], [3.0, 4.0], [5.0, 6.0],  # First mask
            [7.0, 8.0], [9.0, 10.0], [11.0, 12.0]  # Second mask
        ])  # Shape: [6, 2]

        # Expected reshaped and normalized scores
        expected_all_scores = tf.constant([
            [4.0, 5.0],  # (1+7)/2, (2+8)/2
            [6.0, 7.0],  # (3+9)/2, (4+10)/2
            [8.0, 9.0]  # (5+11)/2, (6+12)/2
        ])  # Shape: [3, 2]

        result = self.scorer.reshape_and_normalize(all_scores, mask_count, batch_size)

        self.assertTrue(tf.reduce_all(tf.equal(result, expected_all_scores)))


if __name__ == "__main__":
    unittest.main()
