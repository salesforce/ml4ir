## Using custom feature transformation functions

ml4ir allows users to define custom feature transformation functions. Here, we demonstrate how to define a function to convert text into character embeddings and then encode using a bidirectional GRU.

```
def bytes_sequence_to_encoding_bilstm(feature_tensor, feature_info, file_io: FileIO):
    args = feature_info["feature_layer_info"]["args"]

    # Decode string tensor to bytes
    feature_tensor = io.decode_raw(
        feature_tensor, out_type=tf.uint8, fixed_length=args.get("max_length", None),
    )

    feature_tensor = tf.squeeze(feature_tensor, axis=1)
    if "embedding_size" in args:
        char_embedding = layers.Embedding(
            name="{}_bytes_embedding".format(
                feature_info.get("node_name", feature_info.get("name"))
            ),
            input_dim=256,
            output_dim=args["embedding_size"],
            mask_zero=True,
            input_length=args.get("max_length", None),
        )(feature_tensor)
    else:
        char_embedding = tf.one_hot(feature_tensor, depth=256)

    kernel_initializer = args.get("lstm_kernel_initializer", "glorot_uniform")
    encoding = get_bigru_encoding(
        embedding=char_embedding,
        lstm_units=int(args["encoding_size"] / 2),
        kernel_initializer=kernel_initializer,
    )
    return encoding


def get_bigru_encoding(embedding, lstm_units, kernel_initializer="glorot_uniform"):
    encoding = layers.Bidirectional(
        layers.GRU(
            units=lstm_units, return_sequences=False, kernel_initializer=kernel_initializer
        ),
        merge_mode="concat",
    )(embedding)
    encoding = tf.expand_dims(encoding, axis=1)
    return encoding
```
**Note:** Any feature transformation function has to be a tensorflow compatible function as it is part of the tensorflow-keras `RelevanceModel`.


Define the feature transformation function to use with a text feature like query text:
```
- name: query_text
  node_name: query_text
  trainable: true
  dtype: string
  log_at_inference: true
  feature_layer_info:
    fn: bytes_sequence_to_encoding_bigru
    args:
      encoding_type: bilstm
      encoding_size: 128
      embedding_size: 128
      max_length: 20
  serving_info:
    name: query_text
    required: true
```

Finally, use the custom transformation functions with the `InteractionModel` and consecutively, create a `RelevanceModel`:
```
custom_feature_transform_fns = {
    "bytes_sequence_to_encoding_bigru": bytes_sequence_to_encoding_bigru,
}

interaction_model: InteractionModel = UnivariateInteractionModel(
                                            feature_config=feature_config,
                                            feature_layer_keys_to_fns=custom_feature_transform_fns,
                                            tfrecord_type=TFRecordTypeKey.EXAMPLE,
                                            file_io=file_io)
```

Once the `InteractionModel` has been wrapped with a `Scorer`, metrics, etc we can define a `RelevanceModel`. This model can be used for training, prediction and evaluation.