## Serving ml4ir Models on the JVM

ml4ir provides Scala utilities for serving a saved model in a JVM based production environment. The utilities provide an easy way to use the `FeatureConfig` used at training time to map serving features from a production feature store or Solr into model features. These model features can then be packaged as a TFRecord protobuf message, which is then fed into the model. The utilities fetch the scores returned from the model which can then be used as necessary. For example, the scores can be used by the JVM code to

* convert ranking scores to ranks for each document per query

* sort documents based on ranking scores for each document

* convert classification scores to top label

and so on.


#### A high level usage of the Scala utilities

Load the FeatureConfig, saved model and create handlers to convert raw serving time features into TFRecord protos
```
val featureConfig = ModelFeaturesConfig.load(featureConfigPath)
val sequenceExampleBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(featureConfig)
val rankingModelConfig = ModelExecutorConfig(inputTFNode, scoresTFNode)
val rankingModel = new SavedModelBundleExecutor(modelPath, rankingModelConfig)
```

Load serving time features from a CSV file. Replace this step with any other production feature store or Solr

```
val queryContextsAndDocs = StringMapCSVLoader.loadDataFromCSV(csvDataPath, featureConfig)
```

Convert serving time features into a TFRecord proto message using the FeatureConfig (here, SequenceExample proto)
```
queryContextsAndDocs.map {
case q @ StringMapQueryContextAndDocs(queryContext, docs) =>
  val sequenceExample = sequenceExampleBuilder.build(queryContext, docs)
  (q, sequenceExample, rankingModel(sequenceExample))
}
```

Pass TFRecord protos to the loaded model and fetch ranking scores
```
val allScores: Iterable[
  (StringMapQueryContextAndDocs, SequenceExample, Array[Float])] = runQueriesAgainstDocs(
    pathFor("test_data.csv"),
    pathFor("ranking_model_bundle"),
    pathFor("feature_config.yaml"),
    "serving_tfrecord_protos",
    "ranking_score"
  )
```

Sample returned scores for a query with six documents
```
0.14608994, 0.21464024, 0.1768626, 0.1312356, 0.19536583, 0.13580573
```