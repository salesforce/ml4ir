"""Tests for calibration implemented in `ml4ir.base.model.calibrations.temperature_scaling` """
import shutil
import os

import pandas as pd
import tensorflow as tf
import numpy as np

from ml4ir.base.model.calibration.temperature_scaling import dict_to_zipped_csv, \
    TEMPERATURE_SCALE, accuracy, get_logits_labels, temperature_scale
from ml4ir.applications.classification.tests.test_base import ClassificationTestBase


class TestCalibration(ClassificationTestBase):
    """Class to test temperature scaling from `ml4ir.base.model.calibration.temperature_scaling` """

    def test_dict_to_zipped_csv(self):
        """Tests if the .zip file has been created and contains .csv file.
         It also tests if .csv file contains true values"""

        data_dict = {'feature': [1, 2.0, 3], 'labels': [0, 0, 1]}
        dict_to_zipped_csv(data_dict, self.output_dir, self.file_io)

        filename_zip = os.path.join(self.output_dir, TEMPERATURE_SCALE)
        filename_csv = os.path.join(self.output_dir, TEMPERATURE_SCALE, f'{TEMPERATURE_SCALE}.csv')

        shutil.unpack_archive(f'{filename_zip}.zip', self.output_dir, "zip")
        self.assertTrue(os.path.exists(filename_csv))

        df = pd.read_csv(filename_csv)
        pd.testing.assert_frame_equal(df, pd.DataFrame.from_dict(data_dict))

    def test_accuracy(self):
        """Tests accuracy"""
        logits = np.array([[0.3, 0.2, 0.5], [0.7, 0.2, 0.1]])
        labels = np.array([2, 1])
        scores_tensor = tf.constant(logits, name='logits_test', dtype=tf.float32)
        labels_tensor = tf.constant(labels, name='logits_test', dtype=tf.float32)

        acc = accuracy(scores_tensor, labels_tensor)
        self.assertEqual(acc, 0.5)

    def test_temperature_scaling(self):
        """Tests temperature scaling """

        # computing logits before temperature scaling of the validation set
        logits_numpys, labels_numpys = get_logits_labels(self.classification_model.model,
                                                         self.relevance_dataset.validation)
        # sanity check the shape of logits and labels
        self.assertTrue(len(logits_numpys) == len(labels_numpys))

        results = temperature_scale(self.classification_model.model,
                                    self.classification_model.scorer,
                                    self.relevance_dataset,
                                    self.logger,
                                    self.output_dir, 1.5, self.file_io)

        # Tests the learned temperature parameter
        expected_value = 1.55
        atol = 0.1
        self.assertTrue(np.isclose(results.position[0], expected_value, atol=atol))

        # computing logits after temperature scaling of the validation set
        logits_numpys_w_ts, labels_numpys_w_ts = get_logits_labels(self.classification_model.model,
                                                                   self.relevance_dataset.validation)

        # Tests if the model weights were frozen and not affected by temperature scaling
        np.testing.assert_array_equal(logits_numpys, logits_numpys_w_ts)
        np.testing.assert_array_equal(labels_numpys, labels_numpys_w_ts)

        # Tests the accuracy of the validation set before and after temperature scaling
        logits_tensor = tf.constant(logits_numpys, dtype='float32')
        labels_tenosr = tf.constant(labels_numpys, dtype='int32')
        acc_org = accuracy(logits_tensor, labels_tenosr)

        logits_w_ts_tensor = tf.constant(logits_numpys_w_ts, dtype='float32')
        labels_w_ts_tenosr = tf.constant(labels_numpys_w_ts, dtype='int32')
        acc_ts = accuracy(logits_w_ts_tensor, labels_w_ts_tenosr)

        np.testing.assert_array_equal(acc_org, acc_ts)
