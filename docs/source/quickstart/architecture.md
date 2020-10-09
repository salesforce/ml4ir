## ml4ir's Architecture

ml4ir is designed as a network of tightly coupled modular subcomponents. This lends itself to high customizability. In this section, we will briefly describe each of the subcomponents and how it all fits together.

![](/_static/ml4ir_framework.png)

**FeatureConfig**

The `FeatureConfig` is the main driver of ml4ir bridging the gap between the training and serving side. It is loaded from a YAML file and can be used to configure the list of features used by the model for training and serving. Additionally, it can be used to define preprocessing and feature transformation functions and their respective arguments. It can be extended to configure additional metadata for the features as needed.

More details about defining a FeatureConfig for your ml4ir model **[here](/quickstart/feature_config)**

**Inputs**

Keras Input placeholders constructed from the `FeatureConfig` that forms the first layer of the `RelevanceModel` and will be used by the model for learning a scoring function. Additionally, metadata features can also be made available in the Input layer that can be used to compute custom losses and metrics.

**InteractionModel**

The `InteractionModel` defines the feature transformation layer that converts the Input layer into numeric tensors that can be used to learn a scoring function. This layer can be used for a variety of transformations. Few examples are:

* converting categorical text labels into embedding vectors

* converting text into character embeddings and sequence encoding via a variety of layers like LSTM, GRU, transformers, etc.

* contextual embedding layers such as BERT, ELMO, GPT, etc.

Currently, ml4ir supports a univariate interaction model where a transformation function can be applied to a single feature. This can be extended to define custom interaction models that allow for cross feature interaction based transformations.

**Loss**

`Loss` is an implementation of the `RelevanceLossBase` that can be used to define the loss function and the corresponding final activation layer to be used to train a `RelevanceModel`. The loss function is defined on `y_true` and `y_pred`, the labels and predicted scores from the model, respectively. Metadata features can be used to define complex and custom loss functions to be used with `RelevanceModel`.

**ModelConfig**

`ModelConfig` is a YAML configuration file that defines the scoring function of the `RelevanceModel`. Specifically, it defines the logic to convert the transformed feature vectors into the model score. Currently, the ModelConfig only supports a DNN(multi layer perceptron like) architecture, but can be extended to handle sequential and convolution based scoring functions.

**Scorer**

`Scorer` defines the tensorflow layers of the model to convert the Input layer to the scores by combining and wrapping together the `ModelConfig`, `InteractionModel` and the `Loss`. Custom scorer objects can be defined and used with ml4ir as needed.

**Callbacks**

A list of keras Callbacks that can be used with the `RelevanceModel` for training and evaluation. ml4ir already comes packaged with commonly used callbacks for model checkpointing, early stopping and tensorboard. Additionally, ml4ir also defines debugging callbacks to log training and evaluation progress. Users have the flexibililty to use custom callback functions with ml4ir models as well.

**Optimizer**

Tensorflow's keras based optimizer object that is used for gradient optimization and learning the model weights. ml4ir also plays well with a wide variety of optimizers with custom learning rate schedules such as exponential decay, cyclic learning rate, etc.

**Metrics**

List of keras `Metric` classes that can be used to compute validation and test metrics for evaluating the model. Metadata features can be used to define custom and complex metrics to be used with `RelevanceModel`. 

**RelevanceModel**

The `Scorer` is wrapped with the keras callbacks, optimizer and metrics to define a `RelevanceModel`. The `RelevanceModel` can be used like a Keras model with `fit()`, `predict()`, `evaluate()`, `save()` and `load()` which allow training, evaluation of models for search applications. Pretrained models and weights can be loaded for fine tuning or computing test metrics.

To learn more about constructing a `RelevanceModel` from the ground up check **[this guide](/quickstart/library)**