from __future__ import annotations

from typing import List, Dict

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from ml4ir.base.config.keys import FeatureTypeKey
from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO

tf.compat.v1.enable_resource_variables()


def get_layer_subclasses():
    """
    Get mapping of {subclass-name: subclass} for all derivative classes of keras.Layer
    """

    def full_class_name(cls):
        module = cls.__module__
        if module == 'builtins':
            return cls.__qualname__  # avoid outputs like 'builtins.str'
        return f"{module}.{cls.__qualname__}"

    def get_sub_classes(cls):
        """DFS traversal to get all subclasses of `cls` class"""
        for subcls in cls.__subclasses__():
            if subcls not in subclasses:
                # Handles duplicates
                subclasses[full_class_name(subcls)] = subcls
                get_sub_classes(subcls)

    subclasses = {}
    get_sub_classes(layers.Layer)
    return subclasses


class CycleFoundException(Exception):
    pass


class LayerNode:
    def __init__(
            self,
            name: str,
            layer: keras.Layer = None,
            dependent_children: List[LayerNode] = None,
            inputs: List[str] = None,
            inputs_as_list: bool = False
    ):
        self.name = name
        self.layer = layer
        self.dependent_children = dependent_children if dependent_children is not None else []
        self.inputs = inputs
        self.is_input_node = True if inputs is None else False
        self.inputs_as_list = inputs_as_list

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.__str__()


class LayerGraph:
    def __init__(self, layer_ops, inputs):
        self.nodes = self.create_nodes(layer_ops, inputs)
        self.create_dependency_graph(layer_ops)
        output_nodes = self.get_output_nodes()
        if len(output_nodes) > 1:
            raise NotImplementedError(f"Only 1 output node expected, found {len(output_nodes)}: {output_nodes}")
        self.output_node = output_nodes[0]

    @staticmethod
    def create_nodes(layer_ops, input_names) -> Dict[str, LayerNode]:
        return {
            **{
                name: LayerNode(
                    name=name,
                    layer=layer_op[DenseModel.OP_IDENTIFIER],
                    inputs=layer_op[DenseModel.INPUTS],
                    inputs_as_list=layer_op[DenseModel.INPUTS_AS_LIST]
                ) for name, layer_op in layer_ops.items()
            },
            **{name: LayerNode(name=name) for name in input_names}
        }

    def create_dependency_graph(self, layer_ops):
        for layer_name, layer_op in layer_ops.items():
            for input_node in layer_op[DenseModel.INPUTS]:
                self.nodes[input_node].dependent_children.append(self.nodes[layer_name])

    def get_output_nodes(self) -> List[LayerNode]:
        return list(filter(lambda node: len(node.dependent_children) == 0, self.nodes.values()))

    def topological_sort(self):
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
                dfs(node)

        return order[::-1]


class DenseModel(keras.Model):
    """Dense Model architecture that dynamically maps features -> logits"""
    INPUTS = "inputs"
    LAYER_OP_NAME = "layer_op_name"
    INPUTS_AS_LIST = "aslist"
    LEVELS_IDENTIFIER = "levels"
    LAYERS_IDENTIFIER = "layers"
    LAYER_TYPE = "type"
    NAME = "name"
    OP_IDENTIFIER = "op"

    def __init__(self,
                 model_config: dict,
                 feature_config: FeatureConfig,
                 file_io: FileIO,
                 **kwargs):
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

        # Get all available layers (including keras layers)
        self.available_layers = get_layer_subclasses()
        model_graph = self.define_architecture(model_config)
        self.execution_order: List[LayerNode] = model_graph.topological_sort()
        print("Execution order: ", self.execution_order)
        self.output_node = model_graph.output_node

    def instantiate_op(self, current_layer_type, current_layer_args):
        """
        Create and return an instance of `current_layer_type` with `current_layer_args` params
        """
        try:
            return self.available_layers[current_layer_type](**current_layer_args)
        except KeyError:
            self.file_io.logger.error("Layer type: '%s' is not supported or not found in %s",
                                      current_layer_type, str(self.available_layers))
            raise KeyError(f"Layer type: '{current_layer_type}' "
                           f"is not supported or not found in previous levels")

    def get_layer_op(self, layer_args):
        """Get all the layers for this level"""
        print(layer_args)
        return {
            self.INPUTS: layer_args[self.INPUTS],
            self.OP_IDENTIFIER: self.instantiate_op(layer_args[self.LAYER_TYPE],
                                                    {k: v for k, v in layer_args.items()
                                                     # Exclude items which aren't layer params
                                                     if k not in {self.LAYER_TYPE, self.INPUTS_AS_LIST, self.INPUTS}}),
            self.INPUTS_AS_LIST: layer_args.get(self.INPUTS_AS_LIST, False)
        }

    def define_architecture(self, model_config: dict):
        """
        Convert the model from model_config to a LayerGraph

        :param model_config: dict corresponding to the model config
        """

        layer_ops = {layer_args[self.NAME]: self.get_layer_op(layer_args) for layer_args in
                     model_config[self.LAYERS_IDENTIFIER]}
        inputs = set([input_name for layer_op in layer_ops.values()
                      for input_name in layer_op[self.INPUTS] if input_name not in layer_ops.keys()])
        return LayerGraph(layer_ops, inputs)

    # def build(self, input_shape):
    #     """Build the DNN model"""
    #     self.train_features = sorted(input_shape[FeatureTypeKey.TRAIN])

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

        self.outputs = {k: v for k, v in train_features.items()}

        # Pass features through all the layers of the Model
        for node in self.execution_order:
            # Input nodes don't need any execution
            if node.is_input_node:
                if node.name not in train_features:
                    raise KeyError(f"Input feature {node.name} cannot be found in the feature ops outputs")
            else:
                layer_input = {k: self.outputs[k] for k in node.inputs}
                if node.inputs_as_list or len(layer_input) == 1:
                    layer_input = list(layer_input.values())
                    if len(layer_input) == 1:
                        layer_input = layer_input[0]
                self.outputs[node.name] = node.layer(layer_input, training=training)
                # tf.print(f"node: {node.name} with inputs: {node.inputs} [{node.layer}] has output:",
                #          tf.shape(outputs[node.name]))

        # Collapse extra dimensions
        output_layer = self.output_node
        model_output = self.outputs[output_layer.name]
        if isinstance(output_layer.layer, layers.Dense) and (output_layer.layer.units == 1):
            scores = tf.squeeze(model_output, axis=-1)
        else:
            scores = model_output

        return scores
