"""
Implementing temperature scaling proposed in https://github.com/gpleiss/temperature_scaling
The tensorflow implementation and functions are inspired by
https://www.tensorflow.org/probability/examples/Optimizers_in_TensorFlow_Probability
"""
import sys
import os
import contextlib
import functools
import time
import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
from logging import Logger
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.scoring.scoring_model import ScorerBase

import shutil

TEMPERATURE_SCALE = 'temp_scaling_scores'


def make_val_and_grad_fn(fn):
    """ Returns a function that computes y = fn(*args, **kwargs) and gradients of y wrt
    each of args and  kwargs
    parameters:
        fn : Python callable to be differentiated
    """
    @functools.wraps(fn)
    def val_and_grad(x):
        return tfp.math.value_and_gradient(fn, x)
    return val_and_grad


@contextlib.contextmanager
def timed_execution(logger):
    """ a context manager to compute the execution time of a process usually within
     a 'with' statement
    Parameter:
        logger: logging.Logger object
    """
    t0 = time.time()
    yield
    dt = time.time() - t0
    logger.info(f'Evaluation took: {dt} seconds')


def np_value(tensor):
    """Get numpy value out of possibly nested tuple of tensors.
    Parameters:
        tensor: TF.Tensor
                a tensor object
    """
    if isinstance(tensor, tuple):
        return type(tensor)(*(np_value(t) for t in tensor))
    else:
        return tensor.numpy()


def run(optimizer, logger):
    """Run an optimizer and measure it's evaluation time."""
    optimizer()  # Warmup.
    with timed_execution(logger):
        result = optimizer()
    return np_value(result)


def accuracy(scores, labels):
    """
    returns accuracy
    Parameters:
        scores: tf.Tensor object that contains softmaxes
        labels: tf.Tensor object that contains labels

    """
    acc = tf.keras.metrics.Accuracy()
    acc.update_state(labels, tf.argmax(scores, axis=-1))
    return acc.result().numpy()


def dict_to_zipped_csv(data_dic, root_dir):
    """
    saves input dictionary to a zipped csv file
    Parameters:
        data_dic: input dict to be converted to  zipped csv file
        path: path to save the output file
    """
    # creating zip dir
    zip_dir_path = os.path.join(root_dir, TEMPERATURE_SCALE)
    os.mkdir(zip_dir_path)
    csv_path = os.path.join(zip_dir_path, f'{TEMPERATURE_SCALE}.csv')

    # creating .csv & .zip files
    pd.DataFrame.from_dict(data_dic).to_csv(csv_path, index=False)
    shutil.make_archive(zip_dir_path, "zip", root_dir, TEMPERATURE_SCALE)

    # removing the dir, keeping only the zip
    shutil.rmtree(zip_dir_path)
    return zip_dir_path


def get_intermediate_model(model, scorer):
    """
    returns a tf.keras.models.Model copy of the relavence_model.model. This model must generate
    logits (inputs of softmax).
    Parameters:
        model:
        scorer:
    """
    # get  last layer's output  --> MUST **NOT** BE AN ACTIVATION (e.g. SOFTMAX) LAYER
    final_layer_name = scorer.model_config['layers'][-1]['name']
    layer_output = model.get_layer(name=final_layer_name).output

    return tf.keras.models.Model(inputs=model.input, outputs=layer_output)


def eval_relevance_model(scorer: ScorerBase, logits, labels, temperature=None):
    """
    evaluate the relevance model given the logits and labels
    Parameters:
        scorer: Scorer of the RelevanceModel
        logits: numpy array, logits (input of softmax)
        labels: numpy array, class labels
        temperature: TF.Tensor, a parameter of size=(1)
    Returns:
        accuracy and NLL loss
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
    """
    predicts model output on the given evaluation set
    Returns:
        model output: numpy.array
        labels: numpy.array
    """
    logits_nps = model.predict(evaluation_set).squeeze()
    labels_nps = np.concatenate(list(map(lambda x: x[1], evaluation_set))).squeeze().argmax(axis=-1)
    return logits_nps, labels_nps


def temperature_scale(model: tf.keras.Model, scorer: ScorerBase, dataset: RelevanceDataset,
                      logger: Logger,
                      logs_dir_local: str, temperature_init: float):
    """
    learns a temperature parameter using Temperature Scaling (TS) technique on the validation set.
    It, then, computes the probability scores of the test set with and without TS and writes them
    in a .zip file.
    Parameters:
        model : tf.keras.Model object to be used for temperature scaling
        scorer:  ScorerBase object of the RelevanceModel
        dataset: RelevanceDataset to be used for temperature scaling.
        logger : Logger
        logs_dir_local: path to save the TS scores
        temperature_init: temperature initial value
    """
    # get an intermediate model with logits as output --> MUST NOT BE AN ACTIVATION (e.g.
    # SOFTMAX) LAYER
    intermediate_model = get_intermediate_model(model, scorer)

    start = tf.Variable(initial_value=[temperature_init], shape=(1), dtype=tf.float32, name="temp",
                        trainable=True)

    def nll_loss_fn(temperature):
        """
        computes Negative Log Likelihood loss by applying temperature scaling.
        Parameters:
            temperature: TF.Tensor,  a  parameter of size=(1) to be trained.
        """
        logits_tensor = tf.constant(logits_nps, name='logits_tensor', dtype=tf.float32)
        labels_tensor = tf.constant(labels_nps, name='labels_tensor', dtype=tf.int32)
        logits_w_temp = tf.divide(logits_tensor, temperature)

        return tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_tensor,
                                                           logits=logits_w_temp))

    @tf.function
    def nll_with_lbfgs():
        return tfp.optimizer.lbfgs_minimize(make_val_and_grad_fn(nll_loss_fn),
                                            initial_position=start)

    # evaluation on validation set before temperature scaling
    logits_nps, labels_nps = get_logits_labels(intermediate_model, dataset.validation)
    org_acc_op, org_nll_loss_op, _ = eval_relevance_model(scorer, logits_nps,
                                                          labels_nps)

    # perform temperature scaling
    results = run(nll_with_lbfgs, logger)

    # evaluation on validation set with temperature scaling
    temper = tf.constant(results.position)
    acc_op, nll_loss_op, _ = eval_relevance_model(scorer, logits_nps, labels_nps,
                                                  temperature=temper)

    logger.info("=" * 50)
    logger.info(f'temperature value : {results.position}')
    logger.info('Evaluating on the validation dataset ')
    logger.info(f'original loss: {org_nll_loss_op}, accuracy: {org_acc_op}, \n'
          f'temperature scaling loss: {results.objective_value}, accuracy: {acc_op}\n')

    logger.info("="*50)
    logger.info("Evaluating on the test dataset")

    logits_nps_test, labels_nps_test = get_logits_labels(intermediate_model, dataset.test)

    # evaluation on test set before temperature scaling
    acc_test_ts, _, org_softmaxes = eval_relevance_model(scorer, logits_nps_test,
                                          labels_nps_test)

    # evaluation on test set with temperature scaling
    acc_test_org, _, ts_softmaxes = eval_relevance_model(scorer,
                                                         logits_nps_test,
                                           labels_nps_test, temperature=temper)

    # write the full vector in the csv not ...
    np.set_printoptions(
        formatter={'all': lambda x: str(x.decode('utf-8')) if isinstance(x, bytes) else str(x)},
        linewidth=sys.maxsize,
        threshold=sys.maxsize)
    # avoiding .tolist() that is not memory efficient
    data_dic = {'org_scores': [x for x in org_softmaxes.numpy()],
                'ts_scores': [x for x in ts_softmaxes.numpy()],
                'org_predicted_label': tf.argmax(org_softmaxes, axis=-1).numpy(),
                'true_label': labels_nps_test,
                }

    logger.info(f'original test accuracy: {acc_test_org}, \ntemperature scaling '
                     f'test accuracy: {acc_test_ts} \n')

    # the file is big and should be zipped
    zip_dir_path = dict_to_zipped_csv(data_dic, logs_dir_local)
    logger.info(f"Read and zipped  {zip_dir_path}")

