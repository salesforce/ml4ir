## Defining the ModelConfig

The `ModelConfig` is created from a YAML file and defines the scoring layers of the `RelevanceModel`. Specifically, the model config defines the layers to convert the transformed features output by the `InteractionModel` to the scores for the model. 

Currently, ml4ir supports a dense neural network architecture(multi layer perceptron like). Users can define the type of scoring architecture using the `architecture_key`. The layers of the neural network can be defined as a list of configurations using the `layers` attribute. For each layer, define the `type` of tensorflow-keras layer. Then for each layer, we can specify arguments to be passed to the instantiation of the layer. Finally, for each layer, we can specify a name using the `name` attribute.

Below you can see an example model config YAML using a DNN architecture to stack a bunch of dense layers with ReLU activation layers. Additionally, there are also a few dropout layers for regularization in between.

```
architecture_key: dnn
layers:
  - type: dense
    name: first_dense
    units: 256
    activation: relu
  - type: dropout
    name: first_dropout
    rate: 0.0
  - type: dense
    name: second_dense
    units: 64
    activation: relu
  - type: dropout
    name: second_dropout
    rate: 0.0
  - type: dense
    name: final_dense
    units: 1
    activation: null
```