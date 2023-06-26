import unittest
import numpy as np

from ml4ir.base.features.feature_fns.label_processor import StringMultiLabelProcessor


class StringMultiLabelProcessorTest(unittest.TestCase):
    """Tests for ml4ir.base.features.feature_fns.label_processor.StringMultiLabelProcessor"""

    def test_split_and_combine(self):
        """Test the function is splitting and doing a weighted sum configured"""
        string_multi_label_processor = StringMultiLabelProcessor({
            "name": "test",
            "feature_layer_info": {
                "args": {
                    "separator": "-",
                    "num_labels": 2,
                    "binarize": True,
                    "label_weights": [10, 1],
                }
            }
        })

        inputs = [["1-0", "1-1", "0-1"]]
        outputs = string_multi_label_processor(inputs).numpy()
        self.assertTrue(np.equal(outputs, [[10., 11., 1.]]).all())

    def test_split_on_separator(self):
        """Test the function is splitting on separator configured"""
        string_multi_label_processor = lambda separator: StringMultiLabelProcessor({
            "name": "test",
            "feature_layer_info": {
                "args": {
                    "separator": separator,
                    "num_labels": 2,
                    "binarize": True,
                    "label_weights": [10, 1],
                }
            }
        })

        for separator in ["-", ",", "_", ";", "`"]:  # This is a non-exhaustive list of possible separators
            with self.subTest(f"Layer should be able to split on {separator}"):
                inputs = [separator.join(labels) for labels in [["1", "0"], ["1", "1"], ["0", "1"]]]
                outputs = string_multi_label_processor(separator)(inputs).numpy()
                self.assertTrue(np.equal(outputs, [[10., 11., 1.]]).all())

    def test_binarize(self):
        """Test if the layer can convert input multi labels to binary"""
        string_multi_label_processor = lambda binarize: StringMultiLabelProcessor({
            "name": "test",
            "feature_layer_info": {
                "args": {
                    "separator": "-",
                    "num_labels": 2,
                    "binarize": binarize,
                    "label_weights": [10, 1]
                }
            }
        })
        inputs = [["1-0", "1-1", "0-1"], ["5-0", "5-8", "0-5"]]

        with self.subTest("Layer should convert input multi labels to 1s and 0s when binarize is set to True"):
            outputs = string_multi_label_processor(binarize=True)(inputs).numpy()
            self.assertTrue(np.equal(outputs, [[10., 11., 1.], [10., 11., 1.]]).all())

        with self.subTest("Layer should not convert input multi labels to 1s and 0s when binarize is set to False"):
            outputs = string_multi_label_processor(binarize=False)(inputs).numpy()
            self.assertTrue(np.equal(outputs, [[10., 11., 1.], [50., 58., 5.]]).all())

    def test_default_label_weights(self):
        """Test default label weights when no label weights argument is specified"""
        string_multi_label_processor = StringMultiLabelProcessor({
            "name": "test",
            "feature_layer_info": {
                "args": {
                    "separator": "-",
                    "num_labels": 2,
                    "binarize": True
                }
            }
        })

        inputs = [["1-0", "1-1", "0-1"]]
        outputs = string_multi_label_processor(inputs).numpy()
        self.assertTrue(np.equal(outputs, [[1., 2., 1.]]).all())

    def test_test_label_weights(self):
        """Test that the test label weights are applied when specified"""
        string_multi_label_processor = StringMultiLabelProcessor({
            "name": "test",
            "feature_layer_info": {
                "args": {
                    "separator": "-",
                    "num_labels": 2,
                    "binarize": True,
                    "label_weights": [10, 1],
                    "test_label_weights": [1, 1]
                }
            }
        })

        inputs = [["1-0", "1-1", "0-1"]]
        outputs = string_multi_label_processor(inputs).numpy()
        self.assertTrue(np.equal(outputs, [[1., 2., 1.]]).all())

        train_outputs = string_multi_label_processor(inputs, training=True).numpy()
        self.assertFalse(np.equal(outputs, train_outputs).all())
