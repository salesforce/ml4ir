"""Implementing temperature scaling proposed in https://github.com/gpleiss/temperature_scaling
The tensorflow implementation and functions are inspired by
https://www.tensorflow.org/probability/examples/Optimizers_in_TensorFlow_Probability
"""
import sys
import os
import contextlib
import functools
import time
import shutil

from logging import Logger
from collections import Callable
from typing import Union, Tuple

import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_probability as tfp

from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.scoring.scoring_model import ScorerBase
from ml4ir.base.io.file_io import FileIO

TEMPERATURE_SCALE = 'temp_scaling_scores'


def make_val_and_grad_fn(fun: Callable) -> Callable:
    """Returns a function that computes y = fn(*args, **kwargs) and gradients of y wrt
    each of args and kwargs
    Parameters
    ---------
        fun : Callable
             Python callable to be differentiated
    """
    @functools.wraps(fun)
    def val_and_grad(variables):
        return tfp.math.value_and_gradient(fun, variables)
    return val_and_grad


@contextlib.contextmanager
def timed_execution(logger: Logger) -> None:
    """A context manager to compute the execution time of a process usually within
     a 'with' statement
    Parameter
    --------
        logger: logging.Logger object
                Logger object used for logging

    """
    initial_time = time.time()
    yield
    duration = time.time() - initial_time
    logger.info(f'Evaluation took: {duration} seconds')


def np_value(tensor: Union[tf.Tensor, Tuple[tf.Tensor, ...]]) -> Union[np.ndarray,
                                                                       Tuple[np.ndarray, ...]]:
    """Computes numpy value out of possibly nested tuple of tensors.
    Parameters
    ---------
        tensor: Union[tf.Tensor, Tuple[tf.Tensor, ...]]
                a tensor or a tuple of tensors object
    Returns
    ------
        `Union[np.ndarray, Tuple[np.ndarray, ...]`
         numpy value out of input tensor
    """
    if isinstance(tensor, tuple):
        return type(tensor)(*(np_value(t) for t in tensor))
    return tensor.numpy()


def run_optimizer(optimizer: Callable, logger: Logger) -> Tuple[np.ndarray, ...]:
    """Runs an optimizer and measure it's execution time.
    Parameters
    ---------
        optimizer: function
            It is used for optimization
        logger: Logger
            Logger object used for logging
    Returns
    ------
        `np.ndarray`
        output of the optimizer function
    """
    optimizer()  # Warmup.
    with timed_execution(logger):
        result = optimizer()
    return np_value(result)


def accuracy(scores, labels) -> np.ndarray:
    """Computes accuracy.
    Parameters
    ---------
        scores: tf.Tensor
                tf.Tensor object that contains softmaxes
        labels: tf.Tensor
                tf.Tensor object that contains labels
    Returns
    -------
    `np.ndarray`
    accuracy value

    """
    acc = tf.keras.metrics.Accuracy()
    acc.update_state(labels, tf.argmax(scores, axis=-1))
    return acc.result().numpy()


def dict_to_zipped_csv(data_dic: dict, root_dir: str, file_io: FileIO) -> str:
    """Saves input dictionary to a zipped csv file
    Parameters
    ---------
        data_dic: dict
                input dict to be converted to a zipped csv file
        root_dir: str
                path to save the output file
        file_io: FileIO
                file I/O handler objects for reading and writing data
    Returns
    -------
        `str`
         path to the created zip file
    """
    # creating zip dir
    zip_dir_path = os.path.join(root_dir, TEMPERATURE_SCALE)
    file_io.make_directory(zip_dir_path)

    csv_path = os.path.join(zip_dir_path, f'{TEMPERATURE_SCALE}.csv')

    # creating .csv & .zip files
    pd.DataFrame.from_dict(data_dic).to_csv(csv_path, index=False)
    shutil.make_archive(zip_dir_path, "zip", root_dir, TEMPERATURE_SCALE)

    # removing the dir, keeping only the zip
    shutil.rmtree(zip_dir_path)
    return zip_dir_path


def get_intermediate_model(model, scorer) -> tf.keras.models.Model:
    """Creates a tf.keras.models.Model copy of `model`. This intermediate model must generate
    logits (inputs of softmax).

    Parameters
    ----------
        model: tf.keras.models.Model
                Model object to get the intermediate model from
        scorer: ScorerBase
                scorerBase object to get the `model_config` from

    Returns
    ------
    tf.keras.models.Model
    a tf.keras.models.Model copy of the `model`
    """
    # get  last layer's output  --> MUST **NOT** BE AN ACTIVATION (e.g. SOFTMAX) LAYER
    final_layer_name = scorer.model_config['layers'][-1]['name']
    layer_output = model.get_layer(name=final_layer_name).output

    return tf.keras.models.Model(inputs=model.input, outputs=layer_output)


def eval_relevance_model(scorer: ScorerBase, logits: np.ndarray, labels, temperature=None):
    """Evaluates the relevance model given the logits and labels
    Parameters
    ----------
        scorer: ScorerBase
                Scorer of the RelevanceModel
        logits: numpy.ndarray
                input of softmax
        labels: numpy.ndarray
                class labels
        temperature: TF.Tensor
                temperature parameter of size=(1)
    Returns
    -------
        `np.ndarray`
         accuracy value,
         `tf.Tenosr`
         NLL loss,
         `tf.Tensor`
         softmaxes
    """
    # convert to TF.Tensor
    logits_tensor = tf.constant(logits, name='logits_test', dtype=tf.float32)
    labels_tensor = tf.constant(labels, name='labels_test', dtype=tf.int32)

    if temperature:
        logits_tensor = tf.divide(logits_tensor, temperature)
    nll = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_tensor, logits=logits_tensor))

    # getting the softmaxes using the model final activation function
    softmaxes = scorer.final_activation_op(logits_tensor, {})
    return accuracy(softmaxes, labels_tensor), tf.identity(nll), softmaxes


def get_logits_labels(model: tf.keras.Model, evaluation_set: tf.data.TFRecordDataset):
    """Predicts model output on the given evaluation set
    Parameters:
        model: tf.keras.Model to be used for prediction
        evaluation_set: tf.data.TFRecordDataset to be used for prediction
    Returns:
        model output: numpy.ndarray
        labels: numpy.ndarray
    """
    logits_numpys = model.predict(evaluation_set).squeeze()
    labels_numpys = np.concatenate(list(map(lambda x: x[1], evaluation_set))).squeeze().argmax(
        axis=-1)
    return logits_numpys, labels_numpys


def temperature_scale(model: tf.keras.Model,
                      scorer: ScorerBase,
                      dataset: RelevanceDataset,
                      logger: Logger,
                      logs_dir_local: str,
                      temperature_init: float,
                      file_io: FileIO) -> Tuple[np.ndarray, ...]:
    """learns a temperature parameter using Temperature Scaling (TS) technique on the validation set
    It, then, computes the probability scores of the test set with and without TS and writes them
    in a .zip file.

    Parameters
    ----------
        model :  tf.keras.Model
                 Model object to be used for temperature scaling
        scorer:  ScorerBase object
                 ScorerBase object of the RelevanceModel
        dataset: RelevanceDataset
                 RelevanceDataset object to be used for training and evaluating temperature scaling
        logger : Logger
                 Logger object to log events
        logs_dir_local: str
                 path to save the TS scores
        temperature_init: float
                temperature initial value
        file_io:FileIO
                file I/O handler objects for reading and writing data

    Returns
    -------
        `Tuple[np.ndarray, ...]`
        optimizer output containing temperature value learned during temperature scaling
    """
    # get an intermediate model with logits as output --> MUST NOT BE AN ACTIVATION (e.g.
    # SOFTMAX) LAYER
    intermediate_model = get_intermediate_model(model, scorer)

    start = tf.Variable(initial_value=[temperature_init], shape=(1), dtype=tf.float32, name="temp",
                        trainable=True)

    def nll_loss_fn(temperature):
        """Computes Negative Log Likelihood loss by applying temperature scaling.
        Parameters:
            temperature: TF.Tensor,  a  parameter of size=(1) to be trained
        Returns:
            NLL loss value (averaged on total examples)
        """
        logits_tensor = tf.constant(logits_numpys, name='logits_tensor', dtype=tf.float32)
        labels_tensor = tf.constant(labels_numpys, name='labels_tensor', dtype=tf.int32)
        logits_w_temp = tf.divide(logits_tensor, temperature)

        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_tensor,
                                                           logits=logits_w_temp))

    @tf.function
    def nll_with_lbfgs():
        """Returns optimizer function. Inspired by
         https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py#L60"""
        return tfp.optimizer.lbfgs_minimize(make_val_and_grad_fn(nll_loss_fn),
                                            initial_position=start)

    # evaluation on validation set before temperature scaling
    logits_numpys, labels_numpys = get_logits_labels(intermediate_model, dataset.validation)
    original_acc_op, original_nll_loss_op, _ = eval_relevance_model(scorer, logits_numpys,
                                                                    labels_numpys)

    # perform temperature scaling
    results = run_optimizer(nll_with_lbfgs, logger)

    # evaluation on validation set with temperature scaling
    temper = tf.constant(results.position)
    acc_op, _, _ = eval_relevance_model(scorer, logits_numpys, labels_numpys, temperature=temper)

    logger.info("=" * 50)
    logger.info(f'temperature value : {results.position}')
    logger.info('Evaluating on the validation dataset ')
    logger.info(f'original loss: {original_nll_loss_op}, accuracy: {original_acc_op}, \n'
          f'temperature scaling loss: {results.objective_value}, accuracy: {acc_op}\n')
    logger.info("="*50)
    logger.info("Evaluating on the test dataset")

    logits_numpys_test, labels_numpys_test = get_logits_labels(intermediate_model, dataset.test)

    # evaluation on test set before temperature scaling
    acc_test_original, _, original_softmaxes = eval_relevance_model(scorer, logits_numpys_test,
                                                                    labels_numpys_test)

    # evaluation on test set with temperature scaling
    acc_test_temperature_scaling, _, temperature_scaling_softmaxes =\
        eval_relevance_model(scorer, logits_numpys_test, labels_numpys_test, temperature=temper)

    # write the full vector in the csv not ...
    np.set_printoptions(
        formatter={'all': lambda x: str(x.decode('utf-8')) if isinstance(x, bytes) else str(x)},
        linewidth=sys.maxsize,
        threshold=sys.maxsize)
    # avoiding .tolist() that is not memory efficient
    # note: temperature scaling does not change the accuracy as it does not change the maximum. So,
    # the temperature scaling predicted labels must be the same as  original
    # predicted labels: `original_predicted_label`
    data_dic = {'original_scores': [x for x in original_softmaxes.numpy()],
                'temperature_scaling_scores': [x for x in temperature_scaling_softmaxes.numpy()],
                'original_predicted_label': tf.argmax(original_softmaxes, axis=-1).numpy(),
                'true_label': labels_numpys_test,
                }

    logger.info(f'original test accuracy: {acc_test_original}, \ntemperature scaling '
                     f'test accuracy: {acc_test_temperature_scaling} \n')

    # the file is big and should be zipped
    zip_dir_path = dict_to_zipped_csv(data_dic, logs_dir_local, file_io)
    logger.info(f"Read and zipped  {zip_dir_path}")
    return results
