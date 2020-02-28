package ml4ir.inference.tensorflow.data

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
