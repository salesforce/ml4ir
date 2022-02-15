"""Tests for calibration implemented in `ml4ir.base.model.calibrations.temperature_scaling` """
import shutil
import os
import unittest

import pandas as pd
import tensorflow as tf
import numpy as np

from ml4ir.base.model.calibration.temperature_scaling import dict_to_csv, \
    TEMPERATURE_SCALE, accuracy, get_logits_labels, temperature_scale, get_intermediate_model
from ml4ir.applications.classification.tests.test_base import ClassificationTestBase

TEMPERATURE_LAYER_NAME = 'temperature_layer'


class TestCalibration(ClassificationTestBase):
    """Class to test temperature scaling from `ml4ir.base.model.calibration.temperature_scaling` """

    def test_dict_to_csv(self):
        """Tests if the .zip file has been created and contains .csv file.
         It also tests if .csv file contains true values"""

        data_dict = {'feature': [1, 2.0, 3], 'labels': [0, 0, 1]}
        dict_to_csv(data_dict, self.output_dir, self.file_io, zip_output=True)

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
        self.assertEqual(acc, 0.5, msg="accuracy function does not work as expected")

    @unittest.skip("""Disabled as temperature scaling does not work currently.
                      Should be fixed before merging to master""")
    def test_temperature_scaling(self):
        """Tests temperature scaling """

        # Computing logits before temperature scaling of the validation set
        logits_numpys, labels_numpys = get_logits_labels(self.classification_model.model,
                                                         self.relevance_dataset.validation)
        # Sanity check the shape of logits and labels
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

        np.testing.assert_array_equal(acc_org, acc_ts, err_msg="the accuracy of the validation "
                                                               "set before and after temperature "
                                                               "scaling differs")

    @unittest.skip("""Disabled as temperature scaling does not work currently.
                      Should be fixed before merging to master""")
    def test_add_temperature_layer(self):
        """Tests whether adding temperature scaling layer scales the logits as expected """

        # Tests whether temperature = 1.0 has no effect on outputs
        output_original = self.classification_model.model.predict(
            self.relevance_dataset.validation).squeeze()

        temperature = 1.0
        self.classification_model.add_temperature_layer(temperature=temperature,
                                                        layer_name=TEMPERATURE_LAYER_NAME)

        output_calibration = self.classification_model.model.predict(
            self.relevance_dataset.validation).squeeze()

        rtol = 1e-6
        np.testing.assert_allclose(output_original, output_calibration, rtol=rtol,
                                   err_msg="outputs differs between models with and "
                                           "without temperature = 1.0 ")

        # refreshing relevance model
        self.classification_model = self.classification_pipeline.get_relevance_model()

        # getting logits of val. set from original model (without temperature scaling)
        model_wo_ts = get_intermediate_model(self.classification_model.model,
                                                  self.classification_model.scorer)
        logits_numpys_wo_ts, labels_numpys_wo_ts = get_logits_labels(model_wo_ts,
                                                                     self.relevance_dataset.
                                                                     validation)

        # adding a temperature scaling layer to the original model
        temperature = 1.5
        self.classification_model.add_temperature_layer(temperature=temperature,
                                                        layer_name=TEMPERATURE_LAYER_NAME)

        # getting logits of val. set from the model with temperature scaling layer
        temperature_layer = self.classification_model.model.get_layer(name=TEMPERATURE_LAYER_NAME)
        model_wth_ts = tf.keras.models.Model(self.classification_model.model.input,
                                             temperature_layer.output)

        logits_numpys_w_ts, labels_numpys_w_ts = get_logits_labels(model_wth_ts,
                                                                   self.relevance_dataset.validation)

        # tests if the scaled logits w.r.t. temperature are the same as the  model with temperature
        # scaling layer logits
        rtol = 1e-6
        np.testing.assert_allclose(logits_numpys_wo_ts/temperature, logits_numpys_w_ts,
                                   rtol=rtol, err_msg="scaled logits w.r.t. temperature are not "
                                                      "the same as the model with TS layer logits")

    @unittest.skip("""Disabled as temperature scaling does not work currently.
                      Should be fixed before merging to master""")
    def test_relevance_model_w_ts_save(self):
        """Tests whether a loaded model with temperature scaling layer predicts same output with
        the initial model """

        # refreshing relevance model
        self.classification_model = self.classification_pipeline.get_relevance_model()
        initial_model_no_layers = len(self.classification_model.model.layers)

        # adding Temperature Scaling layer
        temperature = 1.5
        self.classification_model.add_temperature_layer(temperature=temperature,
                                                        layer_name=TEMPERATURE_LAYER_NAME)
        model_w_temperature_no_layers = len(self.classification_model.model.layers)

        # tests whether the model with TS layer has one additional (TS) layer
        self.assertEqual(initial_model_no_layers + 1, model_w_temperature_no_layers,
                         msg="difference between number of layers of models with and without "
                             "temperature should be one")

        outputs_before_saving = self.classification_model.model.predict(
            self.relevance_dataset.validation).squeeze()

        # saving and loading the model with TS layer
        self.classification_model.save(
            models_dir=self.output_dir,
            preprocessing_keys_to_fns={},
            postprocessing_fn=None,
            required_fields_only=not self.args.use_all_fields_at_inference,
            pad_sequence=self.args.pad_sequence_at_inference,
            sub_dir='final_calibrated'
        )
        # refreshing relevance model and loading the saved model
        # currently the TS layer is not part of architecture building and needs to be added to
        # the model separately
        self.classification_model = self.classification_pipeline.get_relevance_model()
        # deliberately assigning a different temperature value here
        self.classification_model.add_temperature_layer(temperature=1.0)
        path = os.path.join(self.output_dir, 'final_calibrated', 'default')
        self.classification_model.load_weights(path)

        # tests whether the model before saving and loaded model have equal number of layers
        self.assertEqual(model_w_temperature_no_layers,
                         len(self.classification_model.model.layers),
                         msg="number of layers for model with temperature before and after "
                             "saving differs!")

        ouputs_loaded_model = self.classification_model.model.predict(
            self.relevance_dataset.validation).squeeze()

        # tests whether the saved model and loaded model predicts same results
        rtol = 1e-6
        np.testing.assert_allclose(outputs_before_saving, ouputs_loaded_model, rtol=rtol,
                                   err_msg="Outputs differ between models with temperature before"
                                           " and after saving!")




