package ml4ir.inference.tensorflow.data

import java.util.{Map => JMap}
import java.util

import com.google.common.collect.ImmutableMap

case class StringMapSequenceExampleBuilder(
    modelFeatures: ModelFeaturesConfig,
    floatFns: util.Map[String, util.function.Function[java.lang.Float, java.lang.Float]],
    longFns: util.Map[String, util.function.Function[java.lang.Long, java.lang.Long]],
    strFns: util.Map[String, util.function.Function[java.lang.String, java.lang.String]])
    extends SequenceExampleBuilder[JMap[String, String], JMap[String, String]](
      FeatureProcessors.forStringMaps(modelFeatures, "context", floatFns, longFns, strFns),
      FeatureProcessors.forStringMaps(modelFeatures, "sequence", floatFns, longFns, strFns)
    )

object StringMapSequenceExampleBuilder {
  def withFeatureProcessors(modelFeatures: ModelFeaturesConfig,
                            floatFns: util.Map[String, util.function.Function[java.lang.Float, java.lang.Float]],
                            longFns: util.Map[String, util.function.Function[java.lang.Long, java.lang.Long]],
                            strFns: util.Map[String, util.function.Function[java.lang.String, java.lang.String]]) =
    StringMapSequenceExampleBuilder(
      modelFeatures,
      floatFns: util.Map[String, util.function.Function[java.lang.Float, java.lang.Float]],
      longFns: util.Map[String, util.function.Function[java.lang.Long, java.lang.Long]],
      strFns: util.Map[String, util.function.Function[java.lang.String, java.lang.String]]
    )

  def withFeatureProcessors(modelFeatures: ModelFeaturesConfig) =
    StringMapSequenceExampleBuilder(modelFeatures, ImmutableMap.of(), ImmutableMap.of(), ImmutableMap.of());
}
