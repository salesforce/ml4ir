import unittest
import warnings
import os
from ml4ir.base.io import logging_utils
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.features.preprocessing import *
from ml4ir.base.model.optimizers import cyclic_learning_rate
from ml4ir.base.config.keys import DataFormatKey, TFRecordTypeKey
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.base.model.relevance_model import RelevanceModel
from ml4ir.base.model.losses.loss_base import RelevanceLossBase
from ml4ir.base.model.scoring.scoring_model import ScorerBase, RelevanceScorer
from ml4ir.base.model.scoring.interaction_model import InteractionModel, UnivariateInteractionModel
from ml4ir.base.model.optimizers.optimizer import get_optimizer
from ml4ir.applications.ranking.model.ranking_model import RankingModel
from ml4ir.applications.ranking.config.keys import LossKey
from ml4ir.applications.ranking.config.keys import ScoringTypeKey
from ml4ir.applications.ranking.model.losses import loss_factory
import yaml
from ml4ir.base.features.feature_config import ExampleFeatureConfig, SequenceExampleFeatureConfig
import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras.optimizers.schedules import ExponentialDecay

warnings.filterwarnings("ignore")

INPUT_DIR = "ml4ir/applications/ranking/tests/data/"
MODEL_CONFIG = "ml4ir/applications/ranking/tests/data/configs/model_config_cyclic_lr.yaml"
KEEP_ADDITIONAL_INFO = 0
NON_ZERO_FEATURES_ONLY = 0

class LrCallback(keras.callbacks.Callback):
    """Defining a callback to keep track of changing learning rate during training."""

    def __init__(self):
        self.lr_list = []

    def on_train_batch_begin(self, batch, logs=None):
        self.lr_list.append(self.model.optimizer._decayed_lr(tf.float32).numpy())

    def get_lr_list(self):
        return self.lr_list

class TestLrSchedules(unittest.TestCase):
    """Testing different learning rate schedules"""

    def setUp(self):
        self.feature_config_yaml_convert_to_clicks = INPUT_DIR + \
            'ranklib/feature_config_convert_to_clicks.yaml'
        self.model_config_file = MODEL_CONFIG

    def compare_lr_values(self, scheduler, expected_values):
        """Expects a learning rate scheduler `scheduler` and
        a list of values. Compares ...."""
        actual_values = [scheduler[i].numpy() for i in range(len(expected_values))]
        return np.all(np.isclose(actual_values, expected_values, rtol=0.001))

    def parse_config(self, tfrecord_type: str, feature_config, io) -> SequenceExampleFeatureConfig:
        if feature_config.endswith(".yaml"):
            feature_config = io.read_yaml(feature_config)
        else:
            feature_config = yaml.safe_load(feature_config)

        return SequenceExampleFeatureConfig(feature_config, None)

    def test_cyclic_tri_lr(self):
        """Test a triangular cyclic learning rate"""
        gold = [0.1,0.28000003,0.45999995,0.64000005,0.81999993,1.0,0.81999993,0.64000005,0.45999995,0.28000003,
                0.1,0.28000003,0.46000007,0.6399999,0.81999993,1.0,0.81999993,0.6399999,0.46000007,0.28000003,
                0.1,0.27999982,0.46000007,0.6399999,0.8200002]
        lrs = cyclic_learning_rate.TriangularCyclicalLearningRate(
            initial_learning_rate=0.1,
            maximal_learning_rate=1.0,
            step_size=5,
        )
        scheduler = [lrs(i) for i in range(25)]
        self.assertTrue(self.compare_lr_values(scheduler, gold))

    def test_cyclic_tri2_lr(self):
        """Test a triangular2 cyclic learning rate"""
        gold = [0.1,0.28000003,0.45999995,0.64000005,0.81999993,1.0,0.81999993,0.64000005,0.45999995,0.28000003,0.1,
                0.19000003,0.28000003,0.36999995,0.45999995,0.55,0.45999995,0.36999995,0.28000003,0.19000003,0.1,
                0.14499995,0.19000003,0.23499998,0.28000003]
        lrs = cyclic_learning_rate.Triangular2CyclicalLearningRate(
            initial_learning_rate=0.1,
            maximal_learning_rate=1.0,
            step_size=5,
        )
        scheduler = [lrs(i) for i in range(25)]
        self.assertTrue(self.compare_lr_values(scheduler, gold))

    def test_cyclic_exp_lr(self):
        """Test a exponential cyclic learning rate"""
        gold = [0.1,0.26200002431869507,0.39159995317459106,0.49366000294685364,0.572391927242279,0.6314409971237183,
                0.4826374351978302,0.35828033089637756,0.25496816635131836,0.16973569989204407,0.10000000149011612,
                0.156485915184021,0.2016746699810028,0.23726072907447815,0.2647128999233246,0.2853020131587982,
                0.23341745138168335,0.19005677103996277,0.1540340781211853,0.12431533634662628,0.10000000149011612,
                0.11969540268182755,0.1354517638683319,0.1478598564863205,0.15743185579776764]

        lrs = cyclic_learning_rate.ExponentialCyclicalLearningRate(
            initial_learning_rate=0.1,
            maximal_learning_rate=1.0,
            step_size=5,
            gamma=0.9,
        )
        scheduler = [lrs(i) for i in range(25)]
        self.assertTrue(self.compare_lr_values(scheduler, gold))

    def test_exp_lr(self):
        """Test a exponential learning rate"""
        gold = [0.10000000149011612,0.09486832469701767,0.08999999612569809,0.08538150042295456,0.08099999278783798,
                0.07684334367513657,0.07289998978376389,0.06915900856256485,0.06560999155044556,0.062243103981018066,
                0.05904899165034294,0.05601879581809044,0.05314409360289574,0.050416912883520126,0.04782968387007713,
                0.045375220477581024,0.04304671287536621,0.04083769768476486,0.03874203935265541,0.036753926426172256,
                0.03486783429980278,0.03307853266596794,0.03138105198740959,0.029770677909255028,0.028242945671081543]

        lrs = ExponentialDecay(
            initial_learning_rate=0.1,
            decay_steps=2,
            decay_rate=0.9,
        )
        scheduler = [lrs(i) for i in range(25)]
        self.assertTrue(self.compare_lr_values(scheduler, gold))

    def test_constant_lr(self):
        """Test a constant learning rate"""
        lrs = ExponentialDecay(
            initial_learning_rate=0.1,
            decay_steps=10000000,
            decay_rate=1.0,
        )
        for i in range(25):
            assert round(float(lrs(i).numpy()), 5) == 0.1

    def test_cyclic_lr_in_training_pipeline(self):
        """Test a cyclic learning rate in model training"""
        Logger = logging_utils.setup_logging(
            reset=True,
            file_name=os.path.join(INPUT_DIR + 'ranklib', "output_log.csv"),
            log_to_file=True,
        )

        io = LocalIO()
        feature_config = self.parse_config(TFRecordTypeKey.SEQUENCE_EXAMPLE, self.feature_config_yaml_convert_to_clicks,
                                           io)

        dataset = RelevanceDataset(
            data_dir=INPUT_DIR + '/ranklib',
            data_format=DataFormatKey.RANKLIB,
            feature_config=feature_config,
            tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
            batch_size=2,
            file_io=io,
            preprocessing_keys_to_fns={},
            logger=Logger,
            keep_additional_info=KEEP_ADDITIONAL_INFO,
            non_zero_features_only=NON_ZERO_FEATURES_ONLY,
            max_sequence_size=319,
        )

        # Define interaction model
        interaction_model: InteractionModel = UnivariateInteractionModel(
            feature_config=feature_config,
            feature_layer_keys_to_fns={},
            tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
            max_sequence_size=319,
            file_io=io,
        )

        # Define loss object from loss key
        loss: RelevanceLossBase = loss_factory.get_loss(
            loss_key=LossKey.RANK_ONE_LISTNET, scoring_type=ScoringTypeKey.POINTWISE
        )

        # Define scorer
        scorer: ScorerBase = RelevanceScorer.from_model_config_file(
            model_config_file=self.model_config_file,
            interaction_model=interaction_model,
            loss=loss,
            logger=Logger,
            file_io=io,
        )

        optimizer: Optimizer = get_optimizer(
            model_config=io.read_yaml(self.model_config_file))

        # Combine the above to define a RelevanceModel
        relevance_model: RelevanceModel = RankingModel(
            feature_config=feature_config,
            tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
            scorer=scorer,
            optimizer=optimizer,
            model_file=None,
            file_io=io,
            logger=Logger,
        )
        callbacks_list = []
        my_callback_object = LrCallback()
        callbacks_list.append(my_callback_object)

        history = relevance_model.model.fit(
            x=dataset.train,
            validation_data=dataset.validation,
            epochs=2,
            verbose=True,
            callbacks=callbacks_list,
        )
        lr_list = my_callback_object.get_lr_list()
        lr_gold = [0.001,0.020800006,0.040599994,0.0604,0.080199994,0.1,0.080199994,0.0604,0.040599994,0.020800006,0.001,
                   0.010900003,0.020800006,0.030699994,0.040599994,0.050499998,0.040599994,0.030699994,0.020800006,
                   0.010900003,0.001,0.0059499955,0.010900003,0.015849996,0.020800006,0.02575,0.020800006,0.015849996,
                   0.010900003,0.0059499955,0.001,0.0034749978,0.0059500015,0.008424998,0.010900003,0.013375,
                   0.010900003,0.008424998,0.0059500015,0.0034749978,0.001,0.0022374988,0.0034749978,0.0047125025,
                   0.0059500015,0.0071875,0.0059500015,0.0047125025,0.0034749978, 0.0022374988]

        assert np.all(np.isclose(lr_gold, lr_list))

    def tearDown(self):
        pass

if __name__ == "__main__":
    unittest.main()
