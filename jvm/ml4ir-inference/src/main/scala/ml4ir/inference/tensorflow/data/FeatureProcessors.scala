package ml4ir.inference.tensorflow.data

import java.lang.{Float => JFloat, Long => JLong, String => JString}
import java.util.{Map => JMap}

import scala.collection.JavaConverters._
import java.util.function.{Function => JFunction}

import ml4ir.inference.tensorflow.data.FeaturesConfigHelper._
import org.tensorflow.DataType

object FeatureProcessors {
  val simpleFloatExtractor: String => (JMap[String, String] => Option[Float]) =
    (servingName: String) => (raw: JMap[String, String]) => raw.asScala.get(servingName).map(_.toFloat)
  val simpleLongExtractor: String => (JMap[String, String] => Option[Long]) =
    (servingName: String) => (raw: JMap[String, String]) => raw.asScala.get(servingName).map(_.toLong)
  val simpleStringExtractor: String => (JMap[String, String] => Option[String]) =
    (servingName: String) => (raw: JMap[String, String]) => raw.asScala.get(servingName)

  def toScalaFns(floatFns: JMap[String, JFunction[JFloat, JFloat]],
                 longFns: JMap[String, JFunction[JLong, JLong]],
                 strFns: JMap[String, JFunction[JString, JString]])
    : (Map[String, Float => Float], Map[String, Long => Long], Map[String, String => String]) = {
    val perFeatureFloatFunctions: Map[String, Float => Float] =
      floatFns.asScala.toMap.mapValues(jf => (f: Float) => jf(f).floatValue()).withDefaultValue(identity)
    val perFeatureLongFunctions: Map[String, Long => Long] =
      longFns.asScala.toMap.mapValues(lf => (l: Long) => lf(l).longValue()).withDefaultValue(identity)
    val perFeatureStringFunctions: Map[String, String => String] =
      strFns.asScala.toMap.mapValues(sf => (s: String) => sf(s)).withDefaultValue(identity)
    (perFeatureFloatFunctions, perFeatureLongFunctions, perFeatureStringFunctions)
  }

  def forStringMaps(featuresConfig: FeaturesConfig,
                    tfRecordType: String,
                    floatFns: JMap[String, JFunction[JFloat, JFloat]],
                    longFns: JMap[String, JFunction[JLong, JLong]],
                    strFns: JMap[String, JFunction[JString, JString]]): StringMapFeatureProcessor = {
    val (scalaFloatFns, scalaLongFns, scalaStringFns) = toScalaFns(floatFns, longFns, strFns)
    val perFeaturePrimitiveProcessors: Map[DataType, Map[String, PrimitiveProcessor]] =
      PrimitiveProcessors.fromFunctionMaps(
        featuresConfig,
        tfRecordType,
        scalaFloatFns,
        scalaLongFns,
        scalaStringFns
      )
    StringMapFeatureProcessor(featuresConfig, perFeaturePrimitiveProcessors)
  }

  def forStringMaps(modelFeaturesConfig: ModelFeaturesConfig,
                    tfRecordType: String,
                    floatFns: JMap[String, JFunction[JFloat, JFloat]],
                    longFns: JMap[String, JFunction[JLong, JLong]],
                    strFns: JMap[String, JFunction[JString, JString]]): StringMapFeatureProcessor =
    forStringMaps(modelFeaturesConfig.toFeaturesConfig(tfRecordType), tfRecordType, floatFns, longFns, strFns)
}

case class StringMapFeatureProcessor(featuresConfig: FeaturesConfig,
                                     primitiveProcessors: Map[DataType, Map[String, PrimitiveProcessor]])
    extends FeaturePreprocessor[JMap[String, String]](
      featuresConfig,
      FeatureProcessors.simpleFloatExtractor,
      FeatureProcessors.simpleLongExtractor,
      FeatureProcessors.simpleStringExtractor,
      primitiveProcessors
    )
