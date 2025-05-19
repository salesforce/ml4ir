import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from typing import Dict, List, Optional, Union, Any
import networkx as nx

from ml4ir.base.features.feature_config import FeatureConfig
from ml4ir.base.io.file_io import FileIO
from ml4ir.base.model.architectures.layer_factory import get_keras_layer_subclasses, instantiate_keras_layer


class ComplexDNNLayerKey:
    """Constants for ComplexDNN layer configuration keys"""
    TYPE = "type"
    NAME = "name"
    INPUTS = "inputs"
    BRANCH = "branch"
    BRANCHES = "branches"
    OPERATION = "operation"
    ARGS = "args"
    REQUIRES_MASK = "requires_mask"
    TRAINABLE = "trainable"
    LAYERS = "layers"
    POSITIONAL_BIAS_HANDLER = "positional_bias_handler"


class ComplexDNNOperation:
    """Supported operations for ComplexDNN"""
    # Basic layers
    DENSE = "dense"
    BATCH_NORMALIZATION = "batch_normalization"
    DROPOUT = "dropout"
    ACTIVATION = "activation"
    
    # Convolutional layers
    CONV1D = "conv1d"
    CONV2D = "conv2d"
    MAX_POOL1D = "max_pool1d"
    MAX_POOL2D = "max_pool2d"
    AVG_POOL1D = "avg_pool1d"
    AVG_POOL2D = "avg_pool2d"
    
    # Recurrent layers
    LSTM = "lstm"
    GRU = "gru"
    BIDIRECTIONAL = "bidirectional"
    RNN = "rnn"
    
    # Embedding layers
    EMBEDDING = "embedding"
    
    # Regularization layers
    LAYER_NORMALIZATION = "layer_normalization"
    SPATIAL_DROPOUT = "spatial_dropout"
    
    # Operation layers
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    DIVIDE = "divide"
    DOT = "dot"
    CONCATENATE = "concatenate"
    AVERAGE = "average"
    MAX = "max"
    MIN = "min"
    
    # Reshape operations
    RESHAPE = "reshape"
    FLATTEN = "flatten"
    GLOBAL_AVG_POOL = "global_avg_pool"
    GLOBAL_MAX_POOL = "global_max_pool"
    
    # Advanced layers
    LAMBDA = "lambda"
    RESIDUAL = "residual"
    ATTENTION = "attention"
    TRANSFORMER = "transformer"
    
    # Custom layers
    CUSTOM = "custom"


class ComplexDNN(keras.Model):
    """
    Complex DNN architecture that supports:
    1. Multiple input branches
    2. Complex operations between layers (add, multiply, concatenate, etc.)
    3. Custom layer definitions
    4. Branch merging and splitting
    5. Residual connections
    6. Attention mechanisms
    """

    def __init__(
        self,
        model_config: dict,
        feature_config: FeatureConfig,
        file_io: FileIO,
        **kwargs
    ):
        """
        Initialize a complex DNN architecture

        Parameters
        ----------
        model_config: dict
            Dictionary defining the complex architecture spec
        feature_config: FeatureConfig
            FeatureConfig defining how each input feature is used in the model
        file_io: FileIO
            File input output handler
        """
        super().__init__(**kwargs)

        self.file_io: FileIO = file_io
        self.model_config = model_config
        self.feature_config = feature_config
        self.available_keras_layers = get_keras_layer_subclasses()

        # Create the network graph
        self.network_graph = self._build_network_graph()
        
        # Get execution order using topological sort
        self.execution_order = list(nx.topological_sort(self.network_graph))
        
        # Register all layers with the model
        self._register_layers()
        
        # Store layer outputs for operations
        self.layer_outputs = {}

    def _build_network_graph(self) -> nx.DiGraph:
        """
        Build a directed graph representing the network architecture
        
        Returns
        -------
        nx.DiGraph
            Network graph with layers as nodes and dependencies as edges
        """
        graph = nx.DiGraph()
        
        # Add input nodes
        for input_name in self.feature_config.get_all_features(include_label=False):
            graph.add_node(input_name, type="input")
        
        # Process each layer configuration
        for layer_config in self.model_config[ComplexDNNLayerKey.LAYERS]:
            layer_name = layer_config[ComplexDNNLayerKey.NAME]
            layer_type = layer_config[ComplexDNNLayerKey.TYPE]
            
            # Add layer node
            graph.add_node(layer_name, type=layer_type, config=layer_config)
            
            # Add edges from inputs to this layer
            for input_name in layer_config.get(ComplexDNNLayerKey.INPUTS, []):
                graph.add_edge(input_name, layer_name)
        
        return graph

    def _register_layers(self):
        """Register all layers with the model"""
        for node in self.execution_order:
            if self.network_graph.nodes[node]["type"] != "input":
                layer_config = self.network_graph.nodes[node]["config"]
                layer = self._create_layer(layer_config)
                setattr(self, f"layer_{node}", layer)

    def _create_layer(self, layer_config: dict) -> keras.layers.Layer:
        """
        Create a keras layer from configuration
        
        Parameters
        ----------
        layer_config: dict
            Layer configuration dictionary
            
        Returns
        -------
        keras.layers.Layer
            Created keras layer
        """
        layer_type = layer_config[ComplexDNNLayerKey.TYPE]
        layer_args = layer_config.get(ComplexDNNLayerKey.ARGS, {})
        
        # Basic layers
        if layer_type == ComplexDNNOperation.DENSE:
            return layers.Dense(**layer_args)
        elif layer_type == ComplexDNNOperation.BATCH_NORMALIZATION:
            return layers.BatchNormalization(**layer_args)
        elif layer_type == ComplexDNNOperation.DROPOUT:
            return layers.Dropout(**layer_args)
        elif layer_type == ComplexDNNOperation.ACTIVATION:
            return layers.Activation(**layer_args)
            
        # Convolutional layers
        elif layer_type == ComplexDNNOperation.CONV1D:
            return layers.Conv1D(**layer_args)
        elif layer_type == ComplexDNNOperation.CONV2D:
            return layers.Conv2D(**layer_args)
        elif layer_type == ComplexDNNOperation.MAX_POOL1D:
            return layers.MaxPooling1D(**layer_args)
        elif layer_type == ComplexDNNOperation.MAX_POOL2D:
            return layers.MaxPooling2D(**layer_args)
        elif layer_type == ComplexDNNOperation.AVG_POOL1D:
            return layers.AveragePooling1D(**layer_args)
        elif layer_type == ComplexDNNOperation.AVG_POOL2D:
            return layers.AveragePooling2D(**layer_args)
            
        # Recurrent layers
        elif layer_type == ComplexDNNOperation.LSTM:
            return layers.LSTM(**layer_args)
        elif layer_type == ComplexDNNOperation.GRU:
            return layers.GRU(**layer_args)
        elif layer_type == ComplexDNNOperation.RNN:
            return layers.SimpleRNN(**layer_args)
        elif layer_type == ComplexDNNOperation.BIDIRECTIONAL:
            inner_layer = self._create_layer(layer_args.get("layer", {}))
            return layers.Bidirectional(inner_layer, **{k: v for k, v in layer_args.items() if k != "layer"})
            
        # Embedding layers
        elif layer_type == ComplexDNNOperation.EMBEDDING:
            return layers.Embedding(**layer_args)
            
        # Regularization layers
        elif layer_type == ComplexDNNOperation.LAYER_NORMALIZATION:
            return layers.LayerNormalization(**layer_args)
        elif layer_type == ComplexDNNOperation.SPATIAL_DROPOUT:
            return layers.SpatialDropout1D(**layer_args)
            
        # Operation layers
        elif layer_type == ComplexDNNOperation.ADD:
            return layers.Add(**layer_args)
        elif layer_type == ComplexDNNOperation.SUBTRACT:
            return layers.Subtract(**layer_args)
        elif layer_type == ComplexDNNOperation.MULTIPLY:
            return layers.Multiply(**layer_args)
        elif layer_type == ComplexDNNOperation.DIVIDE:
            return layers.Divide(**layer_args)
        elif layer_type == ComplexDNNOperation.DOT:
            return layers.Dot(**layer_args)
        elif layer_type == ComplexDNNOperation.CONCATENATE:
            return layers.Concatenate(**layer_args)
        elif layer_type == ComplexDNNOperation.AVERAGE:
            return layers.Average(**layer_args)
        elif layer_type == ComplexDNNOperation.MAX:
            return layers.Maximum(**layer_args)
        elif layer_type == ComplexDNNOperation.MIN:
            return layers.Minimum(**layer_args)
            
        # Reshape operations
        elif layer_type == ComplexDNNOperation.RESHAPE:
            return layers.Reshape(**layer_args)
        elif layer_type == ComplexDNNOperation.FLATTEN:
            return layers.Flatten(**layer_args)
        elif layer_type == ComplexDNNOperation.GLOBAL_AVG_POOL:
            return layers.GlobalAveragePooling1D(**layer_args)
        elif layer_type == ComplexDNNOperation.GLOBAL_MAX_POOL:
            return layers.GlobalMaxPooling1D(**layer_args)
            
        # Advanced layers
        elif layer_type == ComplexDNNOperation.LAMBDA:
            return layers.Lambda(**layer_args)
        elif layer_type == ComplexDNNOperation.RESIDUAL:
            return self._create_residual_block(layer_args)
        elif layer_type == ComplexDNNOperation.ATTENTION:
            return self._create_attention_layer(layer_args)
        elif layer_type == ComplexDNNOperation.TRANSFORMER:
            return self._create_transformer_block(layer_args)
            
        # Custom layers
        elif layer_type == ComplexDNNOperation.CUSTOM:
            if "layer_class" not in layer_args:
                raise ValueError("Custom layer must specify 'layer_class' in args")
            layer_class = layer_args.pop("layer_class")
            return instantiate_keras_layer(layer_class, layer_args)
            
        # Try to find the layer in available keras layers
        elif layer_type in self.available_keras_layers:
            return instantiate_keras_layer(layer_type, layer_args)
        else:
            raise ValueError(f"Unsupported layer type: {layer_type}")

    def _create_residual_block(self, layer_args: dict) -> keras.layers.Layer:
        """Create a residual block with skip connection"""
        class ResidualBlock(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.layers = []
                for layer_config in layer_args.get("layers", []):
                    self.layers.append(self._create_layer(layer_config))
                self.add = layers.Add()

            def call(self, inputs):
                x = inputs
                for layer in self.layers:
                    x = layer(x)
                return self.add([inputs, x])

        return ResidualBlock()

    def _create_attention_layer(self, layer_args: dict) -> keras.layers.Layer:
        """Create an attention layer"""
        class AttentionLayer(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.attention_dim = layer_args.get("attention_dim", 64)
                self.W = layers.Dense(self.attention_dim)
                self.b = layers.Dense(1)
                self.softmax = layers.Softmax(axis=1)

            def call(self, inputs):
                # inputs shape: (batch_size, seq_len, feature_dim)
                e = self.W(inputs)  # (batch_size, seq_len, attention_dim)
                e = tf.nn.tanh(e)
                e = self.b(e)  # (batch_size, seq_len, 1)
                a = self.softmax(e)  # (batch_size, seq_len, 1)
                return tf.reduce_sum(a * inputs, axis=1)  # (batch_size, feature_dim)

        return AttentionLayer()

    def _create_transformer_block(self, layer_args: dict) -> keras.layers.Layer:
        """Create a transformer block"""
        class TransformerBlock(keras.layers.Layer):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.embed_dim = layer_args.get("embed_dim", 32)
                self.num_heads = layer_args.get("num_heads", 2)
                self.ff_dim = layer_args.get("ff_dim", 32)
                self.rate = layer_args.get("dropout_rate", 0.1)
                
                self.att = layers.MultiHeadAttention(
                    num_heads=self.num_heads, key_dim=self.embed_dim
                )
                self.ffn = keras.Sequential([
                    layers.Dense(self.ff_dim, activation="relu"),
                    layers.Dense(self.embed_dim),
                ])
                self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
                self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
                self.dropout1 = layers.Dropout(self.rate)
                self.dropout2 = layers.Dropout(self.rate)

            def call(self, inputs, training=False):
                attn_output = self.att(inputs, inputs)
                attn_output = self.dropout1(attn_output, training=training)
                out1 = self.layernorm1(inputs + attn_output)
                ffn_output = self.ffn(out1)
                ffn_output = self.dropout2(ffn_output, training=training)
                return self.layernorm2(out1 + ffn_output)

        return TransformerBlock()

    def call(self, inputs, training=False):
        """
        Forward pass through the network
        
        Parameters
        ----------
        inputs: dict
            Dictionary of input tensors
        training: bool
            Whether the model is in training mode
            
        Returns
        -------
        tensor
            Output tensor
        """
        # Initialize layer outputs with input tensors
        self.layer_outputs = {name: tensor for name, tensor in inputs.items()}
        
        # Process layers in topological order
        for node in self.execution_order:
            if self.network_graph.nodes[node]["type"] != "input":
                layer_config = self.network_graph.nodes[node]["config"]
                layer = getattr(self, f"layer_{node}")
                
                # Get input tensors for this layer
                layer_inputs = [
                    self.layer_outputs[input_name]
                    for input_name in layer_config.get(ComplexDNNLayerKey.INPUTS, [])
                ]
                
                # Apply layer operation
                if len(layer_inputs) == 1:
                    output = layer(layer_inputs[0], training=training)
                else:
                    output = layer(layer_inputs, training=training)
                
                # Store output
                self.layer_outputs[node] = output
        
        # Return the output of the last layer
        return self.layer_outputs[self.execution_order[-1]] 