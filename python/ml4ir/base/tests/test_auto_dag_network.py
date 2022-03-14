import unittest

from tensorflow.keras.layers import Layer

from ml4ir.base.model.architectures.auto_dag_network import LayerNode, get_layer_subclasses, CycleFoundException, \
    LayerGraph


class UserDefinedTestLayerGlobal(Layer):
    pass


class TestGetLayerSubclasses(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.layer_subclasses = get_layer_subclasses()

    def test_local_userdefined_subclass(self):
        """Test to show local classes are also accessible but need to be in scope to be found"""

        class UserDefinedTestLayerLocal(Layer):
            pass

        self.assertNotIn(UserDefinedTestLayerLocal, self.layer_subclasses.values())
        self.assertIn(UserDefinedTestLayerLocal, get_layer_subclasses().values())

    def test_global_userdefined_subclass(self):
        """Test that all globally defined classes are accessible"""
        self.assertIn(UserDefinedTestLayerGlobal, self.layer_subclasses.values())


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
        self.assertListEqual(list(map(str, sorted_order)), ["c", "b", "a", "3", "2", "1"])
