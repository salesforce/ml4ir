package ml4ir.inference.tensorflow.data

case class MultiFeatures(floatFeatures: Map[String, Float] = Map.empty,
                         int64Features: Map[String, Long] = Map.empty,
                         stringFeatures: Map[String, String] = Map.empty,
                         docMetadata: Map[String, String] = Map.empty)
