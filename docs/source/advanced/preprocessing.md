## Using custom preprocessing functions

Preprocessing functions can be used with ml4ir in the data loading pipeline. Below we demonstrate how to define a custom preprocessing function and use it to load the data to train a `RelevanceModel`.

In this example, we define a preprocessing function to split a string into tokens and pad to max length.
```
@tf.function
def split_and_pad_string(feature_tensor, split_char=",", max_length=20):
    tokens = tf.strings.split(feature_tensor, sep=split_char).to_tensor()
    
    padded_tokens = tf.image.pad_to_bounding_box(
        tf.expand_dims(tokens[:, :max_length], axis=-1),
        offset_height=0,
        offset_width=0,
        target_height=1,
        target_width=max_length,
    )    
    padded_tokens = tf.squeeze(padded_tokens, axis=-1)
    
    return padded_tokens
```

Define the preprocessing function in the FeatureConfig YAML:
```
- name: query_text
  node_name: query_text
  trainable: true
  dtype: string
  log_at_inference: true
  preprocessing_info:
    - fn: split_and_pad_string
      args:
        split_char: " "
        max_length: 20
  serving_info:
    name: query_text
    required: true
```

Finally, use the custom split and pad prepreprocessing function to load a `RelevanceDataset` by passing custom functions as the `preprocessing_keys_to_fns` argument:
```
custom_preprocessing_fns = {
    "split_and_pad_string": split_and_pad_string
}

relevance_dataset = RelevanceDataset(
        data_dir=CSV_DATA_DIR,
        data_format=DataFormatKey.CSV,
        feature_config=feature_config,
        tfrecord_type=TFRecordTypeKey.EXAMPLE,
        batch_size=128,
        preprocessing_keys_to_fns=custom_preprocessing_fns,
        file_io=file_io,
        logger=logger
    )
```

Optionally, we can save preprocessing functions in the SavedModel as part of the serving signature as well. This requires that the preprocessing function is a `tf.function` that can be serialized as a tensorflow layer.
```
relevance_model.save(
    models_dir=MODEL_DIR,
    preprocessing_keys_to_fns=custom_preprocessing_fns,
    required_fields_only=True)
```