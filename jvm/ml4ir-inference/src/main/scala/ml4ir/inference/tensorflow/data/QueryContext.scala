package ml4ir.inference.tensorflow.data

case class QueryContext(queryString: String,
                        queryId: String,
                        queryMetadata: Map[String, String] = Map.empty)
    extends CSVWritable {
  override def toCsvString(separator: String): String = {
    import MapToCsvImplicits._
    List(queryId, queryString, queryMetadata.toCsvString(separator))
      .mkString(separator)
  }
}
