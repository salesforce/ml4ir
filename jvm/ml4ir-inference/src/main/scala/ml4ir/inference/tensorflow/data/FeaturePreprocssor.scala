package ml4ir.inference.tensorflow.data

import ml4ir.inference.tensorflow.utils.{FeatureConfig, FeatureField}

/**
  *
  * @param rawInput
  * @tparam T for example: Map[String, String]
  */
abstract class FeaturePreprocssor[T](rawInput: T,
                                     featureConfig: FeatureConfig) {
  def extractFloatFeatures: Map[String, Float]
  def extractLongFeatures: Map[String, Long]
  def extractStringFeatures: Map[String, String]
}

class StringMapFeatureProcessor(input: Map[String, String],
                                featureConfig: FeatureConfig)
    extends FeaturePreprocssor[Map[String, String]](input, featureConfig) {
  override def extractFloatFeatures: Map[String, Float] = ???

  override def extractLongFeatures: Map[String, Long] = ???

  override def extractStringFeatures: Map[String, String] = ???
}
