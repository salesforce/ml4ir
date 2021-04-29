package ml4ir.inference.tensorflow

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.SequenceExampleBuilder
import org.tensorflow.example.SequenceExample

class RankingModelExecutor[Q, S](modelPath: String,
                                 executorConfig: ModelExecutorConfig,
                                 sequenceExampleBuilder: SequenceExampleBuilder[Q, S]) {
  val rankingModel = new TFRecordExecutor(modelPath, executorConfig)

  def apply(queryContext: Q, docs: List[S]): Array[Float] = {
    val sequenceExample: SequenceExample = sequenceExampleBuilder(queryContext, docs)
    val scores: Array[Float] = rankingModel(sequenceExample)
    scores
  }

  def score(queryContext: Q, docs: java.util.List[S]): Array[Float] = apply(queryContext, docs.asScala.toList)
}
