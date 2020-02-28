package ml4ir.inference.tensorflow.data

trait CSVWritable {
  def toCsvString: String = toCsvString(", ")
  def toCsvString(separator: String): String
}

object MapToCsvImplicits {
  implicit class MapToCsv[T](map: Map[String, T]) extends CSVWritable {
    override def toCsvString(separator: String): String = {
      map.keySet.toList.sorted.map(map).map(_.toString).mkString(separator)
    }
  }
}
