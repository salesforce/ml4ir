import math
import unittest
from unittest.mock import MagicMock
import tensorflow as tf
from ml4ir.base.model.scoring.monte_carlo_scorer import MonteCarloScorer
from ml4ir.applications.ranking.model.layers.masking import QueryFeatureMask


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
        self.model_config["monte_carlo_trials"] = {'use_fixed_mask_in_training': True, "use_fixed_mask_in_testing": True}
        self.scorer = MonteCarloScorer(
            model_config=self.model_config,
            feature_config=MagicMock(),
            interaction_model=MagicMock(),
            loss=MagicMock(),
            file_io=MagicMock()
        )

        self.scorer.architecture_op = MagicMock()
        mask = QueryFeatureMask(name="query_feature_mask",
                                 mask_rate=0,
                                 mask_at_inference=True,
                                 use_fixed_masks=True,
                                 requires_mask=True)
        batch_size = 4
        sequence_len = 5
        feature_dim = 3
        f1 = tf.random.uniform((batch_size, sequence_len, feature_dim))
        f2 = tf.random.uniform((batch_size, sequence_len, feature_dim))
        mask(f1)
        inputs = {
            'feature1': tf.constant(f1),
            'feature2': tf.constant(f2)
        }
        result = self.scorer.call(inputs, training=True)
        self.assertEqual(str(result["score"].name).count("iadd"), int(math.pow(2,feature_dim)-1))


if __name__ == "__main__":
    unittest.main()
