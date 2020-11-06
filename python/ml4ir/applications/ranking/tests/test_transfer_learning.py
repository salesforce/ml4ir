import os
import numpy as np
import random
import tensorflow as tf
import glob

from ml4ir.base.config.keys import DataFormatKey
from ml4ir.applications.ranking.config.keys import MetricKey
from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.data.relevance_dataset import RelevanceDataset
from ml4ir.applications.ranking.model.ranking_model import RankingModel
from ml4ir.base.features.feature_config import FeatureConfig


class RankingTransferLearningTest(RankingTestBase):
    """
    This class defines the unit tests around the transfer learning
    capability of RelevanceModel.

    Transfer learning in ml4ir is enabled by saving individual layer weights
    at training time. When defining a new RelevanceModel, these pretrained
    weights can be loaded by passing a mapping of layer name to the path of
    the pretrained weights(saved previously). Additionally, these weights can
    be frozen to avoid fine tuning as needed.
    """

    def get_ranking_dataset_and_model(
        self, seed=123, initialize_layers_dict={}, freeze_layers_list=[]
    ):
        """Helper method to get a RankingModel and Dataset with some default args"""
        data_dir = os.path.join(self.root_data_dir, DataFormatKey.TFRECORD)
        feature_config_path = os.path.join(self.root_data_dir, "configs", self.feature_config_fname)
        data_format = DataFormatKey.TFRECORD
        metrics_keys = [MetricKey.MRR]

        # Fix random seed values for repeatability
        tf.keras.backend.clear_session()
        np.random.seed(seed)
        tf.random.set_seed(seed)
        random.seed(seed)

        feature_config: FeatureConfig = FeatureConfig.get_instance(
            tfrecord_type=self.args.tfrecord_type,
            feature_config_dict=self.file_io.read_yaml(feature_config_path),
            logger=self.logger,
        )

        relevance_dataset = RelevanceDataset(
            data_dir=data_dir,
            data_format=data_format,
            feature_config=feature_config,
            tfrecord_type=self.args.tfrecord_type,
            max_sequence_size=self.args.max_sequence_size,
            batch_size=self.args.batch_size,
            preprocessing_keys_to_fns={},
            train_pcent_split=self.args.train_pcent_split,
            val_pcent_split=self.args.val_pcent_split,
            test_pcent_split=self.args.test_pcent_split,
            use_part_files=self.args.use_part_files,
            parse_tfrecord=True,
            file_io=self.file_io,
            logger=self.logger,
        )

        ranking_model: RankingModel = self.get_ranking_model(
            loss_key=self.args.loss_key,
            feature_config=feature_config,
            metrics_keys=metrics_keys,
            initialize_layers_dict=initialize_layers_dict,
            freeze_layers_list=freeze_layers_list,
        )

        return ranking_model, relevance_dataset

    def test_model_saving(self):
        """
        This unit test checks if the individual layer weights were saved as part of the saved model
        """
        model, dataset = self.get_ranking_dataset_and_model(seed=123)
        model.save(
            models_dir=self.args.models_dir,
            preprocessing_keys_to_fns={},
            postprocessing_fn=None,
            required_fields_only=not self.args.use_all_fields_at_inference,
            pad_sequence=self.args.pad_sequence_at_inference,
        )

        # Check that both default and tfrecord serving signatures have been saved
        saved_directories = [
            os.path.basename(d)
            for d in glob.glob(os.path.join(self.args.models_dir, "final", "*"))
        ]
        assert "default" in saved_directories
        assert "tfrecord" in saved_directories

        # Check if all model layers have been saved
        assert "layers" in saved_directories
        saved_layers = [
            os.path.basename(f)
            for f in glob.glob(os.path.join(self.args.models_dir, "final", "layers", "*"))
        ]
        for layer in model.model.layers:
            assert "{}.npz".format(layer.name) in saved_layers

    def test_loading_pretrained_weights(self):
        """
        This unit test checks if the pretrained weights were loaded into the RankingModel
        """
        pretrained_layer_path = os.path.join(
            self.root_data_dir, "models", "layers", "query_text_bytes_embedding.npz"
        )
        model, dataset = self.get_ranking_dataset_and_model(
            seed=125, initialize_layers_dict={"query_text_bytes_embedding": pretrained_layer_path}
        )

        model_layer_weights = model.model.get_layer("query_text_bytes_embedding").get_weights()
        pretrained_layer_weights = self.file_io.load_numpy_array(pretrained_layer_path)

        for i in range(len(model_layer_weights)):
            assert np.all(model_layer_weights[i] == pretrained_layer_weights[i])

    def test_freeze_weights(self):
        """
        This unit test checks if weights for a layer were frozen at training time
        """
        layer_name = "query_text_bytes_embedding"
        model, dataset = self.get_ranking_dataset_and_model(
            seed=123, freeze_layers_list=[layer_name]
        )

        # Assert non trainable weights
        initial_layer = model.model.get_layer(layer_name)
        assert not initial_layer.trainable
        assert len(initial_layer.trainable_weights) == 0

        # Train the model for 1 epoch and test that the frozen layer weights remain same
        model.fit(dataset=dataset, num_epochs=1, models_dir=self.output_dir)
        final_layer = model.model.get_layer(layer_name)

        for i in range(len(initial_layer.get_weights())):
            assert np.all(initial_layer.get_weights()[i] == final_layer.get_weights()[i])
