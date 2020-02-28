package ml4ir.inference.tensorflow.data

case class RankedQueryDocumentPair(query: QueryContext,
                                   document: Document,
                                   score: Float,
                                   rank: Int)
    extends CSVWritable {
  override def toCsvString(separator: String) =
    List(
      query.toCsvString(separator),
      document.toCsvString(separator),
      String.valueOf(score),
      String.valueOf(rank)
    ).mkString(separator)
}
