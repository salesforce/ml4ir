package ml4ir.inference.tensorflow.data

import org.tensorflow.DataType

/**
  * Wrapper class which uses the ModelFeatures configuration to filter to the allowed features, map from
  * "serving name" to their tensorflow node name, and fill in with defaults when required.
  *
  * Also encapsulates three provided extractor functions from input type T: (t, featureName) => Float, Long, String
  * and turns the whole thing into a feature extractor function: T => Example
  *
  * By construction, this class will have no idea about unknown features which the input T *could* provide, so there
  * is no callback to log this information.  Possible TODO: log features expected by the config, when defaults used.
  *
  * @tparam T for example: Map[String, String]
  */
class FeaturePreprocessor[T](featuresConfig: FeaturesConfig,
                             floatExtractor: String => (T => Option[Float]),
                             longExtractor: String => (T => Option[Long]),
                             stringExtractor: String => (T => Option[String]),
                             primitiveProcessors: Map[DataType, Map[String, PrimitiveProcessor]] =
                               Map.empty.withDefaultValue(Map.empty.withDefaultValue(PrimitiveProcessor())))
    extends (T => Example) {
  val processors: Map[DataType, Map[String, PrimitiveProcessor]] = primitiveProcessors
    .mapValues(_.withDefaultValue(PrimitiveProcessor()))
    .withDefaultValue(Map.empty.withDefaultValue(PrimitiveProcessor()))

  /**
    *
    * @param t object which will have its features extracted into an Example
    * @return the feature-ized Example object
    */
  override def apply(t: T): Example =
    Example.apply(MultiFeatures.apply(extractFloatFeatures(t), extractLongFeatures(t), extractStringFeatures(t)))

  private[this] def extractFloatFeatures(t: T): Map[String, Float] =
    featuresConfig(DataType.FLOAT)
      .map {
        case (servingName, NodeWithDefault(nodeName, defaultValue)) =>
          nodeName -> processors(DataType.FLOAT)(servingName)
            .processFloat(floatExtractor(servingName)(t).getOrElse(defaultValue.toFloat))
      }

  private[this] def extractLongFeatures(t: T): Map[String, Long] =
    featuresConfig(DataType.INT64)
      .map {
        case (servingName, NodeWithDefault(nodeName, defaultValue)) =>
          nodeName -> processors(DataType.INT64)(servingName)
            .processLong(longExtractor(servingName)(t).getOrElse(defaultValue.toLong))
      }
  private[this] def extractStringFeatures(t: T): Map[String, String] =
    featuresConfig(DataType.STRING)
      .map {
        case (servingName, NodeWithDefault(nodeName, defaultValue)) =>
          nodeName -> processors(DataType.STRING)(servingName)
            .processString(stringExtractor(servingName)(t).getOrElse(defaultValue))
      }
}

case class PrimitiveProcessor() {
  def processFloat(f: Float): Float = f
  def processLong(l: Long): Long = l
  def processString(s: String): String = s
}
