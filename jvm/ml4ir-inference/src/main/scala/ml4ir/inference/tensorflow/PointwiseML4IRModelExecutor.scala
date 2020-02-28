package ml4ir.inference.tensorflow

import java.io.PrintWriter
import java.lang

import ml4ir.inference.tensorflow.utils.ModelIO
import ml4ir.inference.tensorflow.data.{
  QueryContext,
  Document,
  RankedQueryDocumentPair
}

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.utils.TensorUtils.{create2Tensor, replicate}
import org.tensorflow.{Graph, Session, Tensor}

class PointwiseML4IRModelExecutor(graph: Graph, config: ModelExecutorConfig)
    extends ((QueryContext, Array[Document]) => Array[Float]) {
  val session = new Session(graph)
  val operations: Set[String] = graph.operations().asScala.map(_.name()).toSet

  /**
    * Score all documents for the supplied query
    * @param query will be truncated and padded to queryLenMax
    * @param documents will be truncated and padded to numDocsPerQuery
    * @return scores array, of length numDocsPerQuery
    */
  override def apply(query: QueryContext,
                     documents: Array[Document]): Array[Float] = {
    val inputTensors: Map[String, Tensor[_]] =
      buildPerDocTensors(documents) + (config.queryNodeName -> buildQueryTensor(
        query
      ))
    try {
      val ranking = Array.ofDim[Float](1, config.numDocsPerQuery)
      inputTensors
        .filterKeys(operations)
        .foldLeft(session.runner()) {
          case (runner, (node, tensor)) => runner.feed(node, tensor)
        }
        .fetch(config.scoresNodeName)
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
      .slice(0, math.min(query.length, config.queryLenMax))
      .getBytes("UTF-8")
      .map(_.floatValue())
      .padTo(config.queryLenMax, 0f)
  }

  /**
    *
    * @param query
    * @return 2-tensor representation of the query, replicating numDocsPerQuery times
    */
  def buildQueryTensor(query: QueryContext): Tensor[_] = {
    create2Tensor(
      replicate(tokenize(query.queryString), config.numDocsPerQuery)
    )
  }

  def buildPerDocTensors(
    docs: Array[Document]
  ): Map[String, Tensor[lang.Float]] = {
    case class FeatureVal(name: String, value: Float, docIdx: Int)
    docs
      .slice(0, math.min(docs.length, config.numDocsPerQuery))
      .zipWithIndex
      .flatMap {
        case (doc: Document, idx: Int) =>
          doc.floatFeatures.map {
            case (feature, value) => FeatureVal(feature, value, idx)
          }
      }
      .groupBy(_.name)
      .mapValues(
        _.sortBy(_.docIdx).map(_.value).padTo(config.numDocsPerQuery, 0f)
      )
      .mapValues(a => create2Tensor(Array(a)))
  }
}

object PointwiseML4IRModelExecutorCLI {

  def run(modelPath: String,
          config: ModelExecutorConfig,
          testSetPath: String,
          outputScorePath: String) = {
    // load up a model executor
    val modelExecutor = new PointwiseML4IRModelExecutor(
      ModelIO.loadTensorflowGraph(modelPath),
      config
    )

    // scored, ranked, and flattened (query, document, score, rank) tuples
    val rankedTestSet: Iterable[RankedQueryDocumentPair] = for {
      (query: QueryContext, docs: Array[Document]) <- loadTestSetIterable(
        testSetPath
      )
      ((doc, score), rank) <- docs
        .zip(modelExecutor(query, docs))
        .sortBy(-_._2)
        .zipWithIndex
    } yield {
      RankedQueryDocumentPair(query, doc, score, rank)
    }

    // write flattened results to back to disk
    val printWriter = new PrintWriter(outputScorePath)
    rankedTestSet.map(_.toCsvString(", ")).foreach(printWriter.println)
    printWriter.close()
  }

  def loadTestSetIterable(
    testSetPath: String
  ): Iterable[(QueryContext, Array[Document])] = ???
}
