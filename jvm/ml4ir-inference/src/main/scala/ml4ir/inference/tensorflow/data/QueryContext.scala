package ml4ir.inference.tensorflow.data

import scala.collection.JavaConverters._

case class QueryContext(features: MultiFeatures)

object QueryContext {
  def apply(floatFeatures: java.util.Map[String, java.lang.Float],
            longFeatures: java.util.Map[String, java.lang.Long],
            stringFeatures: java.util.Map[String, java.lang.String]) =
    new QueryContext(
      MultiFeatures(
        floatFeatures.asScala.mapValues(_.floatValue()).toMap,
        longFeatures.asScala.mapValues(_.longValue()).toMap,
        stringFeatures.asScala.toMap
      )
    )
}
