package ml4ir.inference.tensorflow.data

// FIXME: bring queries in sync with dummy model + config
trait TestData {
  val Q = "q"
  val UID = "userId"

  val baseConfigFile = "model_features_0_0_2.yaml"

  def sampleQueryContexts: List[Map[String, String]] = {
    List(
      Map(Q -> "example query", UID -> "john.smith@example.com"),
      Map(Q -> "another" /* no UID supplied */ ),
      Map( /* no query?!? */ UID -> "user1234")
    )
  }

  def sampleDocumentExamples: List[Map[String, String]] = {
    List(
      Map("docTitle" -> "another document title!", "numDocumentViews" -> "10", "docAgeHours" -> "240"),
      Map("docTitle" -> "The document title!", "docAgeHours" -> "0.5", "numDocumentViews" -> "5"),
      Map("docTitle" -> "", /* "numDocumentViews" -> "10" */ "docAgeHours" -> "24000")
    )
  }
}
