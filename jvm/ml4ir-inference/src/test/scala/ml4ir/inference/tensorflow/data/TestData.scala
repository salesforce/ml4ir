package ml4ir.inference.tensorflow.data

// FIXME: bring queries in sync with dummy model + config
trait TestData {
  val Q = "q"
  val UID = "userId"

  val baseConfigFile = "model_features_0_0_2.yaml"

  def sampleQueryContexts: List[Map[String, String]] = {
    List(
      Map(Q -> "example query", UID -> "john.smith@example.com"),
      Map(Q -> "Another query!" /* no UID supplied */ ),
      Map( /* no query?!? */ UID -> "user1234")
    )
  }

  def sampleDocumentExamples: List[Map[String, String]] = {
    List(
      Map("floatFeat0" -> "0.1", "floatFeat1" -> "0.2", "floatFeat2" -> "0.3"),
      Map("floatFeat0" -> "1.1", "floatFeat1" -> "1.2", "floatFeat2" -> "1.3"),
      Map("floatFeat0" -> "0.01", /* "floatFeat1" -> "10" */ "docAgeHours" -> "24000")
    )
  }
}
