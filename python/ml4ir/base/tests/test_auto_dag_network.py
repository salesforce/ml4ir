import copy
import unittest

import yaml
from ml4ir.base.features.feature_config import SequenceExampleFeatureConfig
from ml4ir.base.io import logging_utils
from ml4ir.base.io.local_io import LocalIO
from ml4ir.base.model.architectures.utils import get_keras_layer_subclasses
from ml4ir.base.model.architectures.auto_dag_network import (LayerNode, CycleFoundException,
                                                             LayerGraph, AutoDagNetwork)
from tensorflow.keras.layers import Layer, Dense


class UserDefinedTestLayerGlobal(Layer):
    pass


class TestGetLayerSubclasses(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_subclasses = get_keras_layer_subclasses()

    def test_local_userdefined_subclass(self):
        """Test to show local classes are also accessible but need to be in scope to be found"""

        class UserDefinedTestLayerLocal(Layer):
            pass

        self.assertNotIn(UserDefinedTestLayerLocal, self.layer_subclasses.values())
        local_layer_subclasses = get_keras_layer_subclasses()
        self.assertIn("ml4ir.base.tests.test_auto_dag_network.TestGetLayerSubclasses.test_local_userdefined_subclass."
                      "<locals>.UserDefinedTestLayerLocal", local_layer_subclasses)
        self.assertIn(UserDefinedTestLayerLocal, local_layer_subclasses.values())

    def test_global_userdefined_subclass(self):
        """Test that all globally defined classes are accessible"""
        self.assertIn("ml4ir.base.tests.test_auto_dag_network.UserDefinedTestLayerGlobal", self.layer_subclasses)
        self.assertIn(UserDefinedTestLayerGlobal, self.layer_subclasses.values())

    def test_keras_native_subclass(self):
        """Test that all globally defined classes are accessible"""
        self.assertIn("keras.layers.core.dense.Dense", self.layer_subclasses)
        self.assertIn(Dense, self.layer_subclasses.values())


class LayerNodeTest(unittest.TestCase):
    def test_is_input_node_population(self):
        self.assertTrue(LayerNode("test_node").is_input_node)
        self.assertFalse(LayerNode("test_node", inputs=["a", "b"]).is_input_node)


class TestLayerGraph(unittest.TestCase):
    def test_exception_on_multiple_outputs(self):
        layer_ops = {
            "1": {
                "inputs": ["c"],
                "op": UserDefinedTestLayerGlobal(),
                "aslist": False
            },
            "2": {
                "inputs": ["a", "b"],
                "op": UserDefinedTestLayerGlobal(),
                "aslist": False
            },
        }
        inputs = ["a", "b", "c"]
        self.assertRaises(NotImplementedError, LayerGraph, layer_ops, inputs)

    def test_exception_on_cyclic_dependency(self):
        with self.subTest("Cycle with no outputs"):
            layer_ops = {
                "1": {
                    "inputs": ["2"],
                    "op": UserDefinedTestLayerGlobal(),
                    "aslist": False
                },
                "2": {
                    "inputs": ["1", "a"],
                    "op": UserDefinedTestLayerGlobal(),
                    "aslist": False
                },
            }
            inputs = ["a", "b", "c"]
            self.assertRaises(CycleFoundException, LayerGraph, layer_ops, inputs)

        with self.subTest("Cycle with outputs"):
            layer_ops = {
                "1": {
                    "inputs": ["2"],
                    "op": UserDefinedTestLayerGlobal(),
                    "aslist": False
                },
                "2": {
                    "inputs": ["3", "a"],
                    "op": UserDefinedTestLayerGlobal(),
                    "aslist": False
                },
                "3": {
                    "inputs": ["1", "c"],
                    "op": UserDefinedTestLayerGlobal(),
                    "aslist": False
                },
                "4": {
                    "inputs": ["1", "2"],
                    "op": UserDefinedTestLayerGlobal(),
                    "aslist": False
                },
            }
            self.assertRaises(CycleFoundException, LayerGraph(layer_ops, inputs).topological_sort)

    def test_simple_topological(self):
        layer_ops = {
            "1": {
                "inputs": ["2"],
                "op": UserDefinedTestLayerGlobal(),
                "aslist": False
            },
            "2": {
                "inputs": ["3", "a"],
                "op": UserDefinedTestLayerGlobal(),
                "aslist": False
            },
            "3": {
                "inputs": ["a", "c"],
                "op": UserDefinedTestLayerGlobal(),
                "aslist": False
            },
        }
        inputs = ["a", "b", "c"]
        sorted_order = LayerGraph(layer_ops, inputs).topological_sort()
        self.assertListEqual(list(map(lambda node: node.name, sorted_order)), ["c", "b", "a", "3", "2", "1"])


class AutoDagNetworkTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_config = yaml.safe_load(
            """
            architecture_key: auto-dag-network
            layers:
              - type: ml4ir.base.tests.test_auto_dag_network.UserDefinedTestLayerGlobal
                name: global1
                inputs:
                  - query_text
                  - text_match_score
                  - page_views_score
                aslist: true
              - type: ml4ir.base.tests.test_auto_dag_network.UserDefinedTestLayerGlobal
                name: global2
                inputs:
                  - Title
                  - text_match_score
                  - page_views_score
                aslist: true
              - type: ml4ir.base.tests.test_auto_dag_network.UserDefinedTestLayerGlobal
                name: global3
                inputs:
                  - text_match_score
                  - page_views_score
                aslist: true
              - type: keras.layers.merging.concatenate.Concatenate
                name: features_concat
                inputs:
                  - global1
                  - global2
                  - global3
                aslist: true
                args:
                  axis: -1
              - type: keras.layers.core.dense.Dense
                name: first_dense
                inputs:
                  - features_concat
                args:
                  units: 512
                  activation: relu
              - type: keras.layers.core.dense.Dense
                name: final_dense
                inputs:
                  - first_dense
                args:
                  units: 1
                  activation: null
            optimizer:
              key: adam
            tie_weights:
              - ["global1", "global2"]
            """
        )
        self.file_io = LocalIO(logging_utils.setup_logging())
        self.feat_config = SequenceExampleFeatureConfig(
            yaml.safe_load(
                """
                query_key: 
                  name: query_id
                  node_name: query_id
                  trainable: false
                  dtype: string
                  log_at_inference: true
                  feature_layer_info:
                    type: numeric
                    shape: null
                  serving_info:
                    name: queryId
                  tfrecord_type: context
                rank:
                  name: rank
                  node_name: rank
                  trainable: false
                  dtype: int64
                  log_at_inference: true
                  feature_layer_info:
                    type: numeric
                    shape: null
                  serving_info:
                    name: originalRank
                    default_value: 0
                  tfrecord_type: sequence
                label:
                  name: clicked
                  node_name: clicked
                  trainable: false
                  dtype: int64
                  log_at_inference: true
                  feature_layer_info:
                    type: numeric
                    shape: null
                  serving_info:
                    name: clicked
                  tfrecord_type: sequence
                features:
                  - name: text_match_score
                    node_name: text_match_score
                    trainable: true
                    dtype: float
                    log_at_inference: true
                    feature_layer_info:
                      type: numeric
                      shape: null
                    serving_info:
                      name: textMatchScore
                    tfrecord_type: sequence
                  - name: page_views_score
                    node_name: page_views_score
                    trainable: true
                    dtype: float
                    log_at_inference: true
                    feature_layer_info:
                      type: numeric
                      shape: null
                      fn: tf_native_op
                      args:
                        ops:
                          - fn: tf.math.add
                            args:
                              y: 0.
                          - fn: tf.math.subtract  # The goal here is to see the end-to-end functionality of tf_native_op without modifying the tests
                            args:
                              y: 0.
                          - fn: tf.clip_by_value
                            args:
                              clip_value_min: 0.
                              clip_value_max: 1000000.
                    serving_info:
                      name: pageViewsScore
                    tfrecord_type: sequence
                  - name: query_text
                    node_name: query_text
                    trainable: true
                    dtype: string
                    log_at_inference: true
                    feature_layer_info:
                      type: numeric
                      shape: null
                      fn: bytes_sequence_to_encoding_bilstm
                      args:
                        encoding_type: bilstm
                        encoding_size: 128
                        embedding_size: 128
                        max_length: 20
                    preprocessing_info:
                      - fn: preprocess_text
                        args:
                          remove_punctuation: true
                          to_lower: true
                    serving_info:
                      name: q
                    tfrecord_type: context
                  - name: Title
                    node_name: Title
                    trainable: true
                    dtype: string
                    log_at_inference: true
                    feature_layer_info:
                      type: numeric
                      shape: null
                      fn: bytes_sequence_to_encoding_bilstm
                      args:
                        encoding_type: bilstm
                        encoding_size: 128
                        embedding_size: 128
                        max_length: 20
                    preprocessing_info:
                      - fn: preprocess_text
                        args:
                          remove_punctuation: true
                          to_lower: true
                    serving_info:
                      name: Title
                    tfrecord_type: context
                """
            ),
            self.file_io.logger
        )

    def test_model_creation(self):
        model = AutoDagNetwork(model_config=self.model_config, feature_config=self.feat_config, file_io=self.file_io)
        first_dense = model.model_graph.get_node("first_dense")
        self.assertEqual(first_dense.layer.units, 512)
        self.assertEqual(first_dense.layer.activation.__name__, "relu")
        final_dense = model.model_graph.get_node("final_dense")
        self.assertEqual(final_dense.layer.units, 1)

    def test_graph_creation(self):
        model = AutoDagNetwork(model_config=self.model_config, feature_config=self.feat_config, file_io=self.file_io)
        graph: LayerGraph = model.model_graph

        def __get_node_name(node_list):
            return [node.name for node in node_list]

        with self.subTest("Check input nodes"):
            graph_input_nodes = __get_node_name(filter(lambda node: node.is_input_node, graph.nodes.values()))
            self.assertSetEqual(set(graph_input_nodes), {"query_text", "text_match_score", "Title", "page_views_score"})
        with self.subTest("Check output node"):
            self.assertEqual(graph.output_node.name, "final_dense")

        with self.subTest("Test graph connections"):
            self.assertListEqual(__get_node_name(graph.get_node("global1").dependent_children), ["features_concat"])
            self.assertListEqual(__get_node_name(graph.get_node("global2").dependent_children), ["features_concat"])
            self.assertListEqual(__get_node_name(graph.get_node("global3").dependent_children), ["features_concat"])
            self.assertListEqual(__get_node_name(graph.get_node("features_concat").dependent_children), ["first_dense"])
            self.assertListEqual(__get_node_name(graph.get_node("first_dense").dependent_children), ["final_dense"])

    def test_tie_weights(self):
        model = AutoDagNetwork(model_config=self.model_config, feature_config=self.feat_config, file_io=self.file_io)
        with self.subTest("Test layers are the same"):
            self.assertEqual(model.model_graph.get_node("global1").layer, model.model_graph.get_node("global2").layer)
        with self.subTest("Inputs should be different"):
            self.assertNotEqual(model.model_graph.get_node("global1").inputs,
                                model.model_graph.get_node("global2").inputs)

    def test_without_tie_weights(self):
        model_config = copy.deepcopy(self.model_config)
        model_config.pop("tie_weights")
        model = AutoDagNetwork(model_config=model_config, feature_config=self.feat_config, file_io=self.file_io)
        self.assertIsNotNone(model.model_graph)

    def test_faulty_tie_weights(self):
        with self.subTest("Single layer not allowed"):
            model_config = copy.deepcopy(self.model_config)
            # Adding an entry with only 1 layer
            model_config["tie_weights"].append(["features_concat"])
            self.assertRaises(ValueError, AutoDagNetwork, model_config, self.feat_config, self.file_io)
        with self.subTest("Layer should be in 1 list only"):
            model_config = copy.deepcopy(self.model_config)
            # Adding another entry with duplicate layer (global1)
            model_config["tie_weights"].append(["global1", "global3"])
            self.assertRaises(ValueError, AutoDagNetwork, model_config, self.feat_config, self.file_io)
        with self.subTest("Incompatible layers"):
            model_config = copy.deepcopy(self.model_config)
            # Adding an entry with incompatible layers
            model_config["tie_weights"].append(["global3", "features_concat"])
            self.assertRaises(TypeError, AutoDagNetwork, model_config, self.feat_config, self.file_io)
