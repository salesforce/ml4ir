package ml4ir.inference.tensorflow.data

// FIXME: bring queries in sync with dummy model + config
trait TestData {
  val Q = "q"
  val UID = "userId"

  val baseConfigFile = "model_features_0_0_2.yaml"

  def sampleQueryContexts: List[Map[String, String]] = {
    List(
      // Map(Q -> "lva3934gv", "domainID" -> "1", "domainName" -> "domain_1")
       Map(Q -> "lva3934gv", "domainName" -> "domain_1")
    )
  }

  def sampleDocumentExamples: List[Map[String, String]] = {
    List(
      Map("textMatchScore" -> "0.6004059401899484", "originalRank" -> "1", "pageViewsScore" -> "0.19435258216121348", "qualityScore" -> "0.30102999566398114", "nameMatch" -> "1"),
      Map("textMatchScore" -> "1.1236655728681209", "originalRank" -> "2", "pageViewsScore" -> "0.19435258216121348", "qualityScore" -> "0.30102999566398114", "nameMatch" -> "1")
    )
  }
}
