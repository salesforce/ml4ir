from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Union

try:
    import pygraphviz as pgv
except ImportError:
    pgv = None

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO
from ml4ir.base.model.architectures.utils import instantiate_keras_layer


class CycleFoundException(Exception):
    """Thrown when a cycle is identified in a DAG"""
    pass


class LayerNode:
    """Defines a node in the `LayerGraph`"""

    def __init__(
            self,
            name: str,
            layer: layers.Layer = None,
            dependent_children: List[LayerNode] = None,
            inputs: List[str] = None,
            inputs_as_list: bool = False
    ):
        """
        Constructor to create a node

        Parameters
        ----------
        name: str
            Name of the node
        layer: instance of type tf.keras.layers.Layer
            Node operation, i.e. Layer for forward prop
        dependent_children: list of LayerNode
            All downstream nodes which depend on this node's output
        inputs: list of str
            All inputs to the `layer` to be passed during forward prop
        inputs_as_list: bool
            Boolean to indicate if the set of input tensors to the layer should be converted to list
        """
        self.name = name
        self.layer = layer
        self.dependent_children = dependent_children if dependent_children is not None else []
        self.inputs = inputs
        self.is_input_node = True if inputs is None else False
        self.inputs_as_list = inputs_as_list

    def __str__(self):
        """Get a readable representation of the node"""
        return f"{self.name}" if self.is_input_node else f"{self.name}\n[{self.layer.__class__.__qualname__}]"


class LayerGraph:
    """Defines a DAG of `LayerNode`"""

    def __init__(
            self,
            layer_ops: Dict[str, Union[str, bool, layers.Layer]],
            inputs: List[str],
            visualization_path: str = None
    ):
        """
        Constructor to create a Graph. It creates a dependency graph and identifies the output node.

        Parameters
        ----------
        layer_ops: dict
            Dictionary which contains information for all the layer nodes in the graph. Expected schema:
            {
                "NodeName":  # Name of the layer node
                {
                    "inputs": ["InputNodeName1", "InputNodeName2"],  # Name of the input layer nodes for this layer
                    "op": keras.layers.Layer.ChildClass(),  # Layer instance which defines the operation of the node
                    "aslist": True  # If inputs to the "op" should be a list of tensors
                }
            }
        inputs: list of str
            All inputs to the model from the interaction model
        """
        self.nodes = self.create_nodes(layer_ops, inputs)
        self.create_dependency_graph()
        output_nodes = self.get_output_nodes()
        if len(output_nodes) == 0:
            raise CycleFoundException("No output nodes found because of cycle in DAG")
        elif len(output_nodes) > 1:
            raise NotImplementedError(
                f"1 output node expected, found {len(output_nodes)}: {list(map(str, output_nodes))}")
        self.output_node = output_nodes[0]

    @staticmethod
    def create_nodes(layer_ops, input_names) -> Dict[str, LayerNode]:
        """
        Create all LayerNodes for the graph

        Parameters
        ----------
        layer_ops: dict
            Dictionary which contains information for all the layer nodes in the graph. Expected schema:
            {
                "name": "NodeName",  # Name of the layer node and layer
                "op": keras.layers.Layer.ChildClass(),  # Layer instance which defines the operation of the node
                "aslist": True  # If inputs to the "op" should be a list of tensors
            }
        input_names: list of str
            All inputs to the model from the interaction model

        Returns
        -------
        Dict[str, LayerNode]
            Dictionary mapping node name to LayerNode instance
        """
        return {
            **{
                name: LayerNode(
                    name=name,
                    layer=layer_op[AutoDagNetwork.OP_IDENTIFIER],
                    inputs=layer_op[AutoDagNetwork.INPUTS],
                    inputs_as_list=layer_op[AutoDagNetwork.INPUTS_AS_LIST]
                ) for name, layer_op in layer_ops.items()
            },
            **{name: LayerNode(name=name) for name in input_names}
        }

    def create_dependency_graph(self):
        """Create a dependency graph using the nodes and corresponding inputs"""
        for node_name, curr_node in self.nodes.items():
            if not curr_node.is_input_node:
                for input_node in curr_node.inputs:
                    self.nodes[input_node].dependent_children.append(curr_node)

    def get_node(self, name: str) -> LayerNode:
        """
        Get a list of all output nodes

        Parameters
        ----------
        name: str
            Name of the node to be retrieved

        Returns
        -------
        LayerNode
            LayerNode instance associated with `name`
        """
        return self.nodes[name]

    def get_output_nodes(self) -> List[LayerNode]:
        """Get a list of all output nodes"""
        # Nodes which do not have outgoing edges (no downstream dependencies), except the feature nodes
        return list(filter(lambda node: (not node.is_input_node) and (len(node.dependent_children) == 0),
                           self.nodes.values()))

    def topological_sort(self) -> List[LayerNode]:
        """
        Get the execution order of nodes such that all inputs to a layer is available for forward prop
        This method also ensures there are no cycles in the DAG. If a cycle is found, throws `CycleFoundException`
        """
        order = []
        path = set()
        visited = set()

        def dfs(node: LayerNode):
            nonlocal order, visited, path
            visited.add(node.name)
            for child in node.dependent_children:
                if child.name in path:
                    raise CycleFoundException("Cycle found in graph. Only DAGs are supported")
                if child.name not in visited:
                    path.add(child.name)
                    dfs(child)
                    path.remove(child.name)
            order.append(node)

        for node in self.nodes.values():
            if node.name not in visited:
                path.add(node.name)
                dfs(node)
                path.remove(node.name)

        return order[::-1]

    def visualize(self, path: str):
        """
        Utility function to visualize the DAG and save DAG image to disk

        Parameters
        ----------
        path: str
            Path to the output visualization file
        """
        vis_graph = pgv.AGraph(directed=True)
        vis_graph.add_edges_from([
            (str(from_node), str(to_node))
            for from_node in self.nodes.values()
            for to_node in from_node.dependent_children
        ])
        vis_graph.draw(path, prog="dot")


class AutoDagNetwork(keras.Model):
    """DAG model architecture dynamically inferred from model config that maps features -> logits"""
    INPUTS = "inputs"
    INPUTS_AS_LIST = "aslist"
    LAYERS_IDENTIFIER = "layers"
    LAYER_TYPE = "type"
    NAME = "name"
    OP_IDENTIFIER = "op"
    LAYER_KWARGS = "args"
    TIED_WEIGHTS = "tie_weights"
    DEFAULT_VIZ_SAVE_PATH = "./"
    GRAPH_VIZ_FILE_NAME = "auto_dag_network.png"

    def __init__(
            self,
            model_config: dict,
            feature_config: FeatureConfig,
            file_io: FileIO,
            **kwargs
    ):
        """
        Initialize a dense neural network layer

        Parameters
        ----------
        model_config: dict
            Dictionary defining the dense architecture spec
        feature_config: FeatureConfig
            FeatureConfig defining how each input feature is used in the model
        file_io: FileIO
            File input output handler
        """
        super().__init__(**kwargs)

        self.file_io: FileIO = file_io
        self.model_config = model_config
        self.feature_config = feature_config

        self.model_graph = self.define_architecture(model_config)

        self.execution_order: List[LayerNode] = self.model_graph.topological_sort()
        # The line below is important for tensorflow to register the available params for the model
        # An alternative is to do this in build()
        # If removed, no layers will be present in the AutoDagNetwork (in the model summary)
        self.register_layers: List[layers.Layer] = [layer_node.layer for layer_node in self.execution_order if
                                                    not layer_node.is_input_node]
        self.file_io.logger.info("Execution order: %s", [node.name for node in self.execution_order])
        self.output_node = self.model_graph.output_node

    def plot_abstract_model(self, plot_dir: str):
        if pgv:
            viz_path = str(Path(plot_dir) / self.GRAPH_VIZ_FILE_NAME)
            self.model_graph.visualize(viz_path)
            self.file_io.log(f"Model DAG visualization can be found here: {viz_path}")
        else:
            self.file_io.log("Skipping visualization. Dependency pygraphviz not found. "
                             "Try installing ml4ir with visualization dependency: pip install ml4ir[visualization]")

    def get_layer_op(
            self,
            layer_args: Dict,
            existing_op: layers.Layer = None
    ) -> Dict[str, Union[str, bool, layers.Layer]]:
        """
        Define the layer operation for a layer in the model config

        Parameters
        ----------
        layer_args: dict
            All the arguments from model config which are needed to define a layer op
        existing_op: instance of type keras.layers.Layer
            If specified, reuse the layer op

        Returns
        -------
        dict
            Dictionary with the layer instance, inputs required for the layer and the format of inputs to the layer
        """
        # Ensure the class names for the current layer and the existing op are the same
        if existing_op and existing_op.__class__.__name__ != layer_args[self.LAYER_TYPE].split(".")[-1]:
            raise TypeError(f"Cannot reuse existing layer of type {existing_op.__class__.__name__}. "
                            f"{layer_args[self.LAYER_TYPE]} is incompatible")
        return {
            self.INPUTS: layer_args[self.INPUTS],
            self.OP_IDENTIFIER: existing_op if existing_op
            else instantiate_keras_layer(layer_args[self.LAYER_TYPE], layer_args.get(self.LAYER_KWARGS, {})),
            self.INPUTS_AS_LIST: layer_args.get(self.INPUTS_AS_LIST, False)
        }

    def define_architecture(self, model_config: dict) -> LayerGraph:
        """
        Convert the model from model_config to a LayerGraph

        Parameters
        ----------
        model_config: dict
            Model config parsed into a dictionary

        Returns
        -------
        LayerGraph
            Dependency DAG for the given model config
        """
        # Handle tied weights
        tied_weights = self.model_config.get(self.TIED_WEIGHTS, [])
        layer_group_mapper = {}
        for group_idx, tied_layers in enumerate(tied_weights):
            if len(tied_layers) < 2:
                raise ValueError(f"Expected at least 2 layers to tie weights, "
                                 f"found {len(tied_layers)} in: {tied_layers}")
            for layer_name in tied_layers:
                if layer_name in layer_group_mapper:
                    raise ValueError(f"Layer {layer_name} found in multiple tied weights lists:\n{tied_weights}")
                layer_group_mapper[layer_name] = group_idx
        group_layer_tracker = {}
        # Get layer ops
        layer_ops = {}
        for layer_args in model_config[self.LAYERS_IDENTIFIER]:
            layer_name = layer_args[self.NAME]
            if tied_weights:
                existing_op = group_layer_tracker.get(layer_group_mapper.get(layer_name, None), None)
                layer_ops[layer_name] = self.get_layer_op(layer_args, existing_op=existing_op)
                if not existing_op and layer_name in layer_group_mapper:
                    group_layer_tracker[layer_group_mapper[layer_name]] = layer_ops[layer_name][self.OP_IDENTIFIER]
            else:
                layer_ops[layer_name] = self.get_layer_op(layer_args)

        # Get all inputs
        # While sorting is not mandatory, it is highly recommended for the sake of reproducibility
        inputs = sorted(set([input_name for layer_op in layer_ops.values()
                             for input_name in layer_op[self.INPUTS] if input_name not in layer_ops.keys()]))
        return LayerGraph(layer_ops, inputs)

    def call(self, inputs, training=None):
        """
        Perform the forward pass for the architecture layer

        Parameters
        ----------
        inputs: dict of dict of tensors
            Input feature tensors divided as train and metadata
        training: bool
            Boolean to indicate if the layer is used in training or inference mode

        Returns
        -------
        tf.Tensor
            Logits tensor computed with the forward pass of the architecture layer
        """
        train_features = inputs[FeatureTypeKey.TRAIN]

        # Do not modify the input
        outputs = {k: v for k, v in train_features.items()}

        # Pass features through all the layers of the Model
        for node in self.execution_order:
            # Input nodes don't need any execution
            if node.is_input_node:
                if node.name not in train_features:
                    raise KeyError(f"Input feature {node.name} cannot be found in the feature ops outputs")
            else:
                # Dict inputs is the default
                layer_input = {k: outputs[k] for k in node.inputs}
                # Handle tensor/list[tensors] as inputs
                if node.inputs_as_list or len(layer_input) == 1:
                    layer_input = list(layer_input.values())
                    # Single input is always sent as a tensor
                    if len(layer_input) == 1:
                        layer_input = layer_input[0]
                outputs[node.name] = node.layer(layer_input, training=training)

        # Collapse extra dimensions
        output_layer = self.output_node
        model_output = outputs[output_layer.name]
        if isinstance(output_layer.layer, layers.Dense) and (output_layer.layer.units == 1):
            scores = tf.squeeze(model_output, axis=-1)
        else:
            scores = model_output

        return scores
