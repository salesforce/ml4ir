## Defining the FeatureConfig

In this section, we describe how to define a feature configuration YAML file for your ml4ir application.

There are two types of feature configs that are supported in ml4ir - `ExampleFeatureConfig` and `SequenceExampleFeatureConfig` corresponding to the two types of TFRecord training and serving data format supported. 

#### Main Keys

The feature config YAML file contains these main keys and their corresponding definitions:

* `query_key` : Feature used to uniquely identify each query (or data point)

* `label` : Feature to be used as the label

* `rank` : Feature to identify the position of the sequence record in a `SequenceExample` proto. It does not need to be specified if using `Example` data format.

* `features` : List of features that are used by the `RelevanceModel` for training and evaluation.

#### Feature Information

For each of the features in the FeatureConfig, we define a corresponding feature information definition. The main keys that should be specified for each feature are:

-----

**`name | str`**

Name of the feature in the input dataset (CSV, TFRecord, libsvm, etc.)

-----

**`node_name | str | default=name`**

Name of the feature in the tensorflow model. This will be the name of the feature in the input layer. Using the same input feature with multiple name nodes and feature transformations is supported. For example, using query text for character and word embeddings.

-----

**`dtype | str`**

Tensorflow data type of the feature. Can be `string`, `int64` or `float`

-----

**`trainable | bool | default=True`**

Value representing whether the feature is to be used for the scoring function. If set to False, the feature is considered a metadata feature that can be used to compute custom metrics and losses. Setting it to True, will make the transformed feature available for scoring by default.

-----

**`tfrecord_type | str`**

Type of the SequenceExample feature type. Can be one of `sequence` for features unique to each sequence record or `context` for features common to all sequence records.

-----

**`preprocessing_info | list of dicts | default=[]`**

List of preprocessing functions to be used on the feature. These functions will be applied in the data loading phase and will not be part of the tensorflow model. ml4ir provides an option to persist preprocessing logic as part of the SavedModel if the preprocessing functions are tensorflow compatible and serializable code.

For each preprocessing function, specify `fn`, the name of the function to be used, and `args`, a dictionary of values that are passed as arguments to the function. For example, to preprocess a text feature to remove punctuation and lower case, one can specify the preprocessing info as below
```
preprocessing_info:
  - fn: preprocess_text
    args:
      remove_punctuation: true
      to_lower: true
```

For more information on defining custom preprocessing functions and using it with ml4ir, check **[this guide](/advanced/preprocessing)**

-----

**`feature_layer_info | dict`**

Definition of the feature transformation function to be applied to the feature in the model. Use this section to specify predefined or custom transformation functions to the model. Only tensorflow compatible functions can be used here as the transformation functions will be part of the `RelevanceModel` and serialized when the model is saved.

To define a feature transformation specify `fn`, the feature transformation function to be applied on the feature, and `args`, the key value pairs to be passed as arguments to the transformation function. For example, to use a text feature to learn character embeddings and produce a sequence encoding by using a bidirectional LSTM, define the feature layer as below
```
feature_layer_info:
  type: numeric
  fn: bytes_sequence_to_encoding_bilstm
  args:
    encoding_type: bilstm
    encoding_size: 128
    embedding_size: 128
    max_length: 20
```

For more information on defining custom feature transformation functions and using it with ml4ir, check **[this guide](/advanced/feature_layer)**

-----

**`serving_info | dict`**

Definition of serving time feature attributes that will be used for model inference in production. Specifically, three key attributes can be specified in this section - `name`, `default_value` and `required`. `name` captures the name of the feature in production feature store that should be mapped to the model feature while constructing the input TFRecord proto. `default_value` captures the value to be used to fill the input feature tensor if the feature is absent in production. `required` is a boolean value representing if the feature is required at inference; the feature tensor will be set to default value otherwise.

-----

**`log_at_inference | bool | default=False`**

Value representing if the feature should be logged when running `RelevanceModel.predict(...)`. Setting to True, returns the feature value when running inference. This can be used for error analysis on test examples and computing more complex metrics in a post processing job.

-----

**`is_group_metric_key | bool | default=False`**

Value representing if the feature should be used for computing groupwise metrics when running `RelevanceModel.evaluate(...)`. The usage and implementation of the groupwise metrics is left to the user to be customized. The Ranking models come prepackaged with groupwise MRR and ACR metrics.

-----

**`is_aux_label | bool | default=False`**

Value representing if the feature is used as an auxiliary label to compute failure metrics and auxiliary loss. The usage of the feature to compute the failure metrics is left to the user to be customized. The Ranking models come prepackaged with failure metrics computation that can be used, for example, to compute rate of clicks on documents without a match on the subject field.

In Ranking applications,

A secondary label is any feature/value that serves as a proxy relevance assessment that the user might be interested to measure on the dataset in addition to the primary click labels. For example, this could be used with an exact query match feature. In that case, the metric sheds light on scenarios where the records with an exact match are ranked lower than those without. This would provide the user with complimentary information (to typical click metrics such as MRR and ACR) about the model to help make better trade-off decisions w.r.t. best model selection.

-----

The `FeatureConfig` can be extended to support additional attributes as necessary.

##### Example

This is an example configuration for the `query_text` feature, which will first be preprocessed to convert to lower case, remove punctuations, etc. Further we transform the feature with a sequence encoding using a bidirectional LSTM. At serving time, the feature `qtext` will be mapped from production feature store into the `query_text` feature for the model.

```
  - name: query_text
    node_name: query_text
    trainable: true
    dtype: string
    log_at_inference: true
    feature_layer_info:
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
      name: qtext
      required: true
      default_value: ""
```
