package ml4ir.inference.tensorflow.data

import java.util

import scala.collection.JavaConverters._
import java.util.function.{Function => JFunction}

import com.google.common.collect.Maps
import ml4ir.inference.tensorflow.data.FeaturesConfigHelper._
import org.tensorflow.DataType

object FeatureProcessors {
  val simpleFloatExtractor: String => (util.Map[String, String] => Option[Float]) =
    (servingName: String) => (raw: java.util.Map[String, String]) => raw.asScala.get(servingName).map(_.toFloat)
  val simpleLongExtractor: String => (util.Map[String, String] => Option[Long]) =
    (servingName: String) => (raw: java.util.Map[String, String]) => raw.asScala.get(servingName).map(_.toLong)
  val simpleStringExtractor: String => (util.Map[String, String] => Option[String]) =
    (servingName: String) => (raw: java.util.Map[String, String]) => raw.asScala.get(servingName)

  def forStringMaps(modelFeatures: ModelFeatures,
                    tfRecordType: String,
                    floatFns: util.Map[String, util.function.Function[java.lang.Float, java.lang.Float]],
                    longFns: util.Map[String, util.function.Function[java.lang.Long, java.lang.Long]],
                    strFns: util.Map[String, util.function.Function[java.lang.String, java.lang.String]])
    : StringMapFeatureProcessor = {
    val featuresConfig = modelFeatures.toFeaturesConfig(tfRecordType)
    val ffns = floatFns.asScala.toMap.withDefaultValue(util.function.Function.identity())
    val lfns = longFns.asScala.toMap.withDefaultValue(util.function.Function.identity())
    val strfns = strFns.asScala.toMap.withDefaultValue(util.function.Function.identity())
    val z: Map[DataType, Map[String, PrimitiveProcessor]] = featuresConfig
      .mapValues(mapping => mapping.withDefaultValue(PrimitiveProcessor()))
      .map {
        case (DataType.FLOAT, nodeMap) =>
          DataType.FLOAT -> nodeMap.map {
            case (servingName, _) =>
              (servingName, new PrimitiveProcessor() {
                override def processFloat(f: Float): Float = ffns(servingName)(f)
              })
          }
        case (DataType.INT64, nodeMap) =>
          DataType.INT64 -> nodeMap.map {
            case (servingName, _) =>
              (servingName, new PrimitiveProcessor() {
                override def processLong(l: Long): Long = lfns(servingName)(l)
              })
          }
        case (DataType.STRING, nodeMap) =>
          DataType.STRING -> nodeMap.map {
            case (servingName, _) =>
              (servingName, new PrimitiveProcessor() {
                override def processString(s: String): String = strfns(servingName)(s)
              })
          }
      }
    StringMapFeatureProcessor(featuresConfig, z)
  }
}

case class StringMapFeatureProcessor(featuresConfig: FeaturesConfig,
                                     primitiveProcessors: Map[DataType, Map[String, PrimitiveProcessor]])
    extends FeaturePreprocessor[java.util.Map[String, String]](
      featuresConfig,
      FeatureProcessors.simpleFloatExtractor,
      FeatureProcessors.simpleLongExtractor,
      FeatureProcessors.simpleStringExtractor,
      primitiveProcessors
    )
