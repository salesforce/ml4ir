package ml4ir.inference.tensorflow.data

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.utils.ModelFeatures
import org.tensorflow.DataType

/**
  *
  * @tparam T for example: Map[String, String]
  */
abstract class FeaturePreprocessor[T](
    modelFeatures: ModelFeatures,
    tfRecordType: String,
    floatExtractor: (T, String) => Option[Float],
    longExtractor: (T, String) => Option[Long],
    stringExtractor: (T, String) => Option[String]
) {
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
  def apply(t: T): Example = Example()

  def extractFloatFeatures(t: T): Map[String, Float] =
    featureNamesByType(DataType.FLOAT)
      .map {
        case (servingName, NodeWithDefault(nodeName, defaultValue)) =>
          nodeName -> floatExtractor(t, servingName).getOrElse(defaultValue.toFloat)
      }

  def extractLongFeatures(t: T): Map[String, Long] =
    featureNamesByType(DataType.INT64)
      .map {
        case (servingName, NodeWithDefault(nodeName, defaultValue)) =>
          nodeName -> longExtractor(t, servingName).getOrElse(defaultValue.toLong)
      }
  def extractStringFeatures(t: T): Map[String, String] =
    featureNamesByType(DataType.STRING)
      .map {
        case (servingName, NodeWithDefault(nodeName, defaultValue)) =>
          nodeName -> stringExtractor(t, servingName).getOrElse(defaultValue)
      }
}

class StringMapFeatureProcessor(modelFeatures: ModelFeatures, tfRecordType: String)
    extends FeaturePreprocessor[Map[String, String]](
      modelFeatures,
      tfRecordType,
      floatExtractor = (rawFeatures: Map[String, String], servingName) => rawFeatures.get(servingName).map(_.toFloat),
      longExtractor = (rawFeatures: Map[String, String], servingName) => rawFeatures.get(servingName).map(_.toLong),
      stringExtractor = (rawFeatures: Map[String, String], servingName) => rawFeatures.get(servingName)
    )
