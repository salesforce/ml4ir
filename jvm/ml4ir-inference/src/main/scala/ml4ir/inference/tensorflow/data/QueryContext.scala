package ml4ir.inference.tensorflow.data

case class QueryContext(queryString: String,
                        queryId: String = QueryContext.DUMMY_QUERY_ID,
                        queryMetadata: Map[String, String] = Map.empty)
    extends CSVWritable {
  override def toCsvString(separator: String): String = {
    import MapToCsvImplicits._
    List(queryId, queryString, queryMetadata.toCsvString(separator))
      .mkString(separator)
  }
}

object QueryContext {
  val DUMMY_QUERY_ID: String = "DUMMY_QUERY_ID"
  def apply(queryString: String) = new QueryContext(queryString)
}
