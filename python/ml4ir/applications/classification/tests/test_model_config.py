import yaml
from ml4ir.applications.classification.tests.test_base import ClassificationTestBase
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.model.architectures.dnn import DNN


class DNNModelTest(ClassificationTestBase):

    def test_dnn_last_layer_units(self):
        feature_config = None

        model_info = yaml.safe_load('''
            architecture_key: dnn
            layers:
              - type: dense
                name: first_dense
                units: 256
                activation: relu
              - type: dense
                name: final_dense
                units: 9
                activation: null
        ''')

        dnn = DNN(model_info, feature_config, self.file_io)
        assert(len(dnn.layer_ops) == 2)

    def test_dnn_last_layer_units_from_vocabulary_file(self):
        feature_config = FeatureConfig(yaml.safe_load('''
            query_key:
              name: query_key
              node_name: query_key
              trainable: false
              dtype: string
            label:
              name: entity_id
              feature_layer_info:
                type: numeric
                fn: categorical_indicator_with_vocabulary_file
                args:
                  vocabulary_file: ml4ir/applications/classification/tests/data/configs/vocabulary/entity_id.csv
            features:
              - name: query_text
                trainable: false
                dtype: string
        '''))

        model_info = yaml.safe_load('''
            architecture_key: dnn
            layers:
              - type: dense
                name: first_dense
                units: 256
                activation: relu
              - type: dense
                name: final_dense
                activation: null
        ''')

        dnn = DNN(model_info, feature_config, self.file_io)
        assert(len(dnn.layer_ops) == 2)

    def test_dnn_drop_out_layers(self):
        feature_config = FeatureConfig(yaml.safe_load('''
            query_key:
              name: query_key
              node_name: query_key
              trainable: false
              dtype: string
            label:
              name: entity_id
              feature_layer_info:
                type: numeric
                fn: categorical_indicator_with_vocabulary_file
                args:
                  vocabulary_file: ml4ir/applications/classification/tests/data/configs/vocabulary/entity_id.csv
            features:
              - name: query_text
                trainable: false
                dtype: string
        '''))

        model_info = yaml.safe_load('''
            architecture_key: dnn
            layers:
              - type: dense
                name: first_dense
                units: 256
                activation: relu
              - type: dropout
                name: first_dropout
                rate: 0.3
              - type: dense
                name: second_dense
                units: 64
                activation: relu
              - type: dropout
                name: second_dropout
                rate: 0.0
              - type: dense
                name: final_dense
                activation: null
        ''')

        dnn = DNN(model_info, feature_config, self.file_io)
        assert(len(dnn.layer_ops) == 5)