package ml4ir.inference.tensorflow

import java.lang

import scala.collection.JavaConverters._

import ml4ir.inference.tensorflow.utils.TensorUtils.{create2Tensor, replicate}
import org.tensorflow.{Graph, Session, Tensor}

class PointwiseML4IRModelExecutor(graph: Graph,
                                  queryNodeName: String,
                                  scoresNodeName: String,
                                  numDocsPerQuery: Int,
                                  queryLenMax: Int) {
  val session = new Session(graph)
  val operations: Set[String] = graph.operations().asScala.map(_.name()).toSet

  /**
    * Score all documents for the supplied query
    * @param query will be truncated and padded to queryLenMax
    * @param documents will be truncated and padded to numDocsPerQuery
    * @return scores array, of length numDocsPerQuery
    */
  def apply(query: String,
            documents: Array[Map[String, Float]]): Array[Float] = {
    val inputTensors: Map[String, Tensor[java.lang.Float]] =
      buildPerDocTensors(documents) + (queryNodeName -> buildQueryTensor(query))
    try {
      val ranking = Array.ofDim[Float](1, numDocsPerQuery)
      inputTensors
        .filterKeys(operations)
        .foldLeft(session.runner()) {
          case (runner, (node, tensor)) => runner.feed(node, tensor)
        }
        .fetch(scoresNodeName)
        .run()
        .get(0)
        .copyTo(ranking)
      ranking(0)
    } finally {
      inputTensors.values.foreach(_.close())
    }
  }

  /**
    *
    * @param query
    * @return UTF-8 encoded byte form of the query (as floats), truncated and padded to queryLenMax
    */
  def tokenize(query: String): Array[Float] = {
    query
      .slice(0, math.min(query.length, queryLenMax))
      .getBytes("UTF-8")
      .map(_.floatValue())
      .padTo(queryLenMax, 0f)
  }

  /**
    *
    * @param query
    * @return 2-tensor representation of the query, replicating numDocsPerQuery times
    */
  def buildQueryTensor(query: String): Tensor[lang.Float] = {
    create2Tensor(replicate(tokenize(query), numDocsPerQuery))
  }

  def buildPerDocTensors(
    docs: Array[Map[String, Float]]
  ): Map[String, Tensor[lang.Float]] = {
    case class FeatureVal(name: String, value: Float, docIdx: Int)
    docs
      .slice(0, math.min(docs.length, numDocsPerQuery))
      .zipWithIndex
      .flatMap {
        case (doc: Map[String, Float], idx: Int) =>
          doc.map { case (feature, value) => FeatureVal(feature, value, idx) }
      }
      .groupBy(_.name)
      .mapValues(_.sortBy(_.docIdx).map(_.value).padTo(numDocsPerQuery, 0f))
      .mapValues(a => create2Tensor(Array(a)))
  }
}
