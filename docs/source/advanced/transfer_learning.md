## Transfer Learning with ml4ir

ml4ir saves individual layer weights as part of the `RelevanceModel.save(...)` call. These layer weights can be used with other ml4ir models for transfer learning. This enables layers like embedding vectors to be shared across search tasks like ranking, classification, etc. with ease.

![serving](/_static/ml4ir_savedmodel.png)

ml4ir provides support for loading pretrained layers and optionally freezing them. Depending on whether these layers/weights need to be fine tuned or used as is, one can freeze these layers or not.

To use pretrained embedding vectors from a `ClassificationModel` on ml4ir with a `RankingModel`:
```
initialize_layers_dict = {
    "query_text_bytes_embedding" : "models/activate_demo/bytes_embedding.npz"
}
freeze_layers_list = ["query_text_bytes_embedding"]
ranking_model: RelevanceModel = RankingModel(
                                    feature_config=feature_config,
                                    tfrecord_type=TFRecordTypeKey.SEQUENCE_EXAMPLE,
                                    scorer=scorer,
                                    metrics=metrics,
                                    optimizer=optimizer,
                                    initialize_layers_dict=initialize_layers_dict,
                                    freeze_layers_list=freeze_layers_list,
                                    file_io=file_io,
                                    logger=logger,
                                )
```

The model can be trained, evaluated and saved as usual after this step.