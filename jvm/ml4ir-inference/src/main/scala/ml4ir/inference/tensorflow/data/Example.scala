package ml4ir.inference.tensorflow.data

import com.google.common.collect.Maps

import scala.collection.JavaConverters._

case class Example(id: String, features: MultiFeatures)

object Example {

  /**
    * Helper builder method from java
    * @param id - maybe not necessary?
    * @param floatFeatures nullable ok
    * @param longFeatures nullable ok
    * @param stringFeatures nullable ok
    * @return fully constructed Example object
    */
  def apply(
    id: String,
    floatFeatures: java.util.Map[String, java.lang.Float],
    longFeatures: java.util.Map[String, java.lang.Long],
    stringFeatures: java.util.Map[String, java.lang.String]
  ): Example = {
    new Example(
      id,
      MultiFeatures(
        Option(floatFeatures)
          .getOrElse(Maps.newHashMap())
          .asScala
          .mapValues(_.floatValue())
          .toMap,
        Option(longFeatures)
          .getOrElse(Maps.newHashMap())
          .asScala
          .mapValues(_.longValue())
          .toMap,
        Option(stringFeatures).getOrElse(Maps.newHashMap()).asScala.toMap
      )
    )
  }
}
