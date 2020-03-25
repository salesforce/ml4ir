package ml4ir.inference.tensorflow.data

import scala.collection.JavaConverters._
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
abstract class FeaturePreprocessor[T](
    modelFeatures: ModelFeatures,
    tfRecordType: String,
    floatExtractor: (T, String) => Option[Float],
    longExtractor: (T, String) => Option[Long],
    stringExtractor: (T, String) => Option[String]
) extends (T => Example) {
  case class NodeWithDefault(nodeName: String, defaultValue: String)
  val featureNamesByType: Map[DataType, Map[String, NodeWithDefault]] =
    modelFeatures.getFeatures.asScala.toList
      .filter(_.getTfRecordType.equalsIgnoreCase(tfRecordType))
      .groupBy(
        inputFeature => DataType.valueOf(inputFeature.getDtype.toUpperCase)
      )
      .mapValues(
        _.map(
          feature => feature.getServingInfo.getName -> NodeWithDefault(feature.getNodeName, feature.getDefaultValue)
        ).toMap
      )
      .withDefaultValue(Map.empty)

  /**
    *
    * @param t object which will have its features extracted into an Example
    * @return the feature-ized Example object
    */
  override def apply(t: T): Example =
    Example.apply(MultiFeatures.apply(extractFloatFeatures(t), extractLongFeatures(t), extractStringFeatures(t)))

  private[this] def extractFloatFeatures(t: T): Map[String, Float] =
    featureNamesByType(DataType.FLOAT)
      .map {
        case (servingName, NodeWithDefault(nodeName, defaultValue)) =>
          // TODO: in case the "orElse" default path is utilized, we could have a callback to log missing features
          nodeName -> floatExtractor(t, servingName).getOrElse(defaultValue.toFloat)
      }

  private[this] def extractLongFeatures(t: T): Map[String, Long] =
    featureNamesByType(DataType.INT64)
      .map {
        case (servingName, NodeWithDefault(nodeName, defaultValue)) =>
          nodeName -> longExtractor(t, servingName).getOrElse(defaultValue.toLong)
      }
  private[this] def extractStringFeatures(t: T): Map[String, String] =
    featureNamesByType(DataType.STRING)
      .map {
        case (servingName, NodeWithDefault(nodeName, defaultValue)) =>
          nodeName -> stringExtractor(t, servingName).getOrElse(defaultValue)
      }
}

class StringMapFeatureProcessor(modelFeatures: ModelFeatures, tfRecordType: String)
    extends FeaturePreprocessor[java.util.Map[String, String]](
      modelFeatures,
      tfRecordType,
      floatExtractor = (rawFeatures: java.util.Map[String, String], servingName) =>
        rawFeatures.asScala.get(servingName).map(_.toFloat),
      longExtractor =
        (rawFeatures: java.util.Map[String, String], servingName) => rawFeatures.asScala.get(servingName).map(_.toLong),
      stringExtractor =
        (rawFeatures: java.util.Map[String, String], servingName) => rawFeatures.asScala.get(servingName)
    )
