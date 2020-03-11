package ml4ir.inference.tensorflow.data

import scala.collection.JavaConverters._

case class Document(docId: String,
                    floatFeatures: Map[String, Float] = Map.empty,
                    int64Features: Map[String, Long] = Map.empty,
                    stringFeatures: Map[String, String] = Map.empty,
                    docMetadata: Map[String, String] = Map.empty)
    extends CSVWritable {
  override def toCsvString(separator: String): String = {
    import MapToCsvImplicits._
    List(
      docId,
      floatFeatures.toCsvString(separator),
      int64Features.toCsvString(separator),
      stringFeatures.toCsvString(separator),
      docMetadata.toCsvString(separator)
    ).mkString(separator)
  }
}

object Document {
  def apply(docId: String,
            floatFeatures: java.util.Map[String, java.lang.Float],
            longFeatures: java.util.Map[String, java.lang.Long],
            stringFeatures: java.util.Map[String, java.lang.String]) = {
    new Document(
      docId,
      floatFeatures.asScala.mapValues(_.floatValue()).toMap,
      longFeatures.asScala.mapValues(_.longValue()).toMap,
      stringFeatures.asScala.toMap
    )
  }
}
