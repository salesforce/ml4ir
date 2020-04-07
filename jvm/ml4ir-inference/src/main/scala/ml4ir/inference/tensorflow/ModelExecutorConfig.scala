package ml4ir.inference.tensorflow

/**
  * TODO: this should be read from the same config file as the {@see ModelFeaturesConfig}
  *
  * @param queryNodeName TensorFlow graph node where the serialized SequenceExample input should go
  * @param scoresNodeName output TensorFlow graph node where scores are retrieved (shape: (1, numInputRecords) )
  */
case class ModelExecutorConfig(queryNodeName: String, scoresNodeName: String)
