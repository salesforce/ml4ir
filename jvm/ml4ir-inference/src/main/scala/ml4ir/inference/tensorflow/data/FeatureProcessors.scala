package ml4ir.inference.tensorflow.data

import java.util

import scala.collection.JavaConverters._
import java.util.function.{Function => JFunction}

import com.google.common.collect.Maps
import ml4ir.inference.tensorflow.data.FeaturesConfigHelper._

object FeatureProcessors {
  /*
  def forStringMapContext(modelFeatures: ModelFeatures) = StringMapFeatureProcessor(modelFeatures, "context")
  def forStringMapContext(modelFeatures: ModelFeatures,
                          tfRecordType: String
                          /*floatFns: FnMap[Float],
                          longFns: FnMap[Long],
                          stringFns: FnMap[String]*/ ) =
    StringMapFeatureProcessor(modelFeatures, "context" /*floatFns, longFns, stringFns*/ )
   */
  val simpleFloatExtractor: (util.Map[String, String], String) => Option[Float] =
    (raw: java.util.Map[String, String], servingName: String) => raw.asScala.get(servingName).map(_.toFloat)
  val simpleLongExtractor: (util.Map[String, String], String) => Option[Long] =
    (raw: java.util.Map[String, String], servingName: String) => raw.asScala.get(servingName).map(_.toLong)
  val simpleStringExtractor: (util.Map[String, String], String) => Option[String] =
    (raw: java.util.Map[String, String], servingName: String) => raw.asScala.get(servingName)
}

case class StringMapFeatureProcessor(modelFeatures: ModelFeatures,
                                     tfRecordType: String,
                                     primitiveProcessors: Map[String, PrimitiveProcessor] =
                                       Map.empty.withDefaultValue(PrimitiveProcessor())
                                     /*,
                                     floatFns: FnMap[Float] = Maps.newHashMap(),
                                     longFns: FnMap[Long] = Maps.newHashMap(),
                                     stringFns: FnMap[String] = Maps.newHashMap()*/ )
    extends FeaturePreprocessor[java.util.Map[String, String]](
      modelFeatures.toFeaturesConfig(tfRecordType),
      FeatureProcessors.simpleFloatExtractor,
      FeatureProcessors.simpleLongExtractor,
      FeatureProcessors.simpleStringExtractor,
      primitiveProcessors
      /*
      floatExtractor = (rawFeatures: java.util.Map[String, String], servingName) => {
        FeatureProcessors
          .simpleFloatExtractor(rawFeatures, servingName)
        //.map(if (servingName == "") dummy else identity /*floatFns.getOrDefault(servingName, JFunction.identity()).apply*/ )
      },
      longExtractor = (rawFeatures: java.util.Map[String, String], servingName) => {
        FeatureProcessors
          .simpleLongExtractor(rawFeatures, servingName)
        //.map(longFns.getOrDefault(servingName, JFunction.identity()).apply)
      },
      stringExtractor = (rawFeatures: java.util.Map[String, String], servingName) => {
        FeatureProcessors
          .simpleStringExtractor(rawFeatures, servingName)
        // .map(stringFns.getOrDefault(servingName, JFunction.identity()).apply)
        rawFeatures.asScala.get(servingName)
      }

     */
    )
