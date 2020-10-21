## Defining the ModelConfig

The `ModelConfig` is created from a YAML file and defines the scoring layers of the `RelevanceModel`. Specifically, the model config defines the layers to convert the transformed features output by the `InteractionModel` to the scores for the model. 

Currently, ml4ir supports a dense neural network architecture (multi layer perceptron like). Users can define the type of scoring architecture using the `architecture_key`. The layers of the neural network can be defined as a list of configurations using the `layers` attribute. For each layer, define the `type` of tensorflow-keras layer. Then for each layer, we can specify arguments to be passed to the instantiation of the layer. Finally, for each layer, we can specify a name using the `name` attribute.

This file is also used to define the optimizer and the learning rate schedule. The current supported optimizers are: `adam`, `adagrad`, `nadam`, `sgd`, `rms_prop`. Each of these optimizers need so set the following hyper-parameter: `gradient_clip_value`. 
The current supported learning rate schedules are: `exponential`, `cyclic` and `constant`. 

The `exponential` learning rate schedule requires defining the following hyper-parameters: `initial_learning_rate`, `decay_steps`, `decay_rate`. For more information, see: https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay

The `cyclic` learning rate schedule has three different type of policies: `triangular`, `triangular2`, `exponential`. All three types require defining the following hyper-parameters: `initial_learning_rate`, `maximal_learning_rate`, `step_size`. The `exponential` type requires and additional hyper-parameter: `gamma`. 
For more information, see: https://www.tensorflow.org/addons/api_docs/python/tfa/optimizers/CyclicalLearningRate and https://arxiv.org/pdf/1506.01186.pdf.

Below you can see an example model config YAML using a DNN architecture to stack a bunch of dense layers with ReLU activation layers. Additionally, there are also a few dropout layers for regularization in between. A triangular2 cyclic learning rate schedule is used with adam optimizer.

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
optimizer: 
  key: adam
  gradient_clip_value: 5.0
lr_schedule:
  key: cyclic
  type: triangular2
  initial_learning_rate: 0.01
  maximal_learning_rate: 0.1
  step_size: 10
```
