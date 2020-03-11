package ml4ir.inference.tensorflow

case class ModelExecutorConfig(queryNodeName: String,
                               scoresNodeName: String,
                               numDocsPerQuery: Int,
                               queryLenMax: Int)
