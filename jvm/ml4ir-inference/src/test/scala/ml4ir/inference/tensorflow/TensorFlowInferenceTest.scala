package ml4ir.inference.tensorflow

import java.io.InputStream

import ml4ir.inference.tensorflow.utils.{ModelIO, SequenceExample4IRBuilder}
import ml4ir.inference.tensorflow.data.{QueryContext, Document}
import org.junit.{Ignore, Test}
import org.junit.Assert._
import org.tensorflow.example._

@Test
class TensorFlowInferenceTest {
  val classLoader = getClass.getClassLoader

  def testQueries: (QueryContext, Array[Document]) = {
    val query = "magic"
    val docsToScore = Array(
      Map(
        "feat_0" -> 0.04f,
        "feat_1" -> 0.08f,
        "feat_2" -> 0.01f,
        "fake_feat" -> 0.2f
      ),
      Map(
        "feat_0" -> 0.4f,
        "feat_1" -> 0.8f,
        "feat_2" -> 0.1f,
        "fake_feat" -> 0.3f
      )
    )
    (
      QueryContext(queryString = query, queryId = "1234Id"),
      docsToScore.zipWithIndex.map {
        case (map, idx) => Document(floatFeatures = map, docId = idx.toString)
      }
    )
  }

  @Test
  def testLoadTFSession = {
    val graphInputStream =
      classLoader.getResourceAsStream("pointwiseModelPointwiseLoss")
    val session = ModelIO.loadTensorflowSession(graphInputStream)
    assertNotNull(session)
  }

  def validateScores(scores: Array[Float], numDocs: Int) = {
    val docScores = scores.take(numDocs)
    val maskedScores = scores.drop(numDocs)
    docScores.foreach(
      score => assertTrue("all docs should score non-negative", score > 0)
    )
    for {
      maskedScore <- maskedScores
      docScore <- docScores
    } {
      assertTrue(
        s"docScore ($docScore) should be > masked score ($maskedScore)",
        docScore > maskedScore
      )
    }
    assertTrue(
      "second doc should score better than first",
      scores(1) > scores(0)
    )
    println(scores.mkString(", "))
  }

  @Test
  def testPointwiseML4IRModelExecutorScoring = {
    val graphInputStream = classLoader.getResourceAsStream("frozen.pb")
    val graph = ModelIO.loadTensorflowGraph(graphInputStream)
    val tfRecordExecutor = new PointwiseML4IRModelExecutor(
      graph = graph,
      ModelExecutorConfig(
        queryNodeName = "query_str",
        scoresNodeName = "ranking_scores/Sigmoid",
        numDocsPerQuery = 25,
        queryLenMax = 20
      )
    )
    val (queryContext, docs) = testQueries
    val scores = tfRecordExecutor(queryContext, docs)
    validateScores(scores, docs.length)
  }

  @Test
  def testSavedModelBundle() = {
    val bundlePath = classLoader.getResource("model_bundle").getPath
    val bundleExecutor = new SavedModelBundleExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_tfrecord_sequence_example_protos",
        scoresNodeName = "StatefulPartitionedCall",
        numDocsPerQuery = 25,
        queryLenMax = 20
      )
    )
    val (queryContext, docs) = testQueries
    val protoBuilder = new SequenceExample4IRBuilder()
    val proto = protoBuilder(queryContext, docs)
    val scores = bundleExecutor(proto)
    validateScores(scores, docs.length)
  }

  @Ignore
  @Test
  def testLoadSequenceExample() = {
    val tfRecordStream: InputStream =
      classLoader.getResourceAsStream("file_0.tfrecord")
    val sequenceExample: SequenceExample =
      SequenceExample.parseFrom(tfRecordStream)
    assertNotNull(sequenceExample)
  }

  @Test
  def testTFRecordInference() = {
    val queryString = "magic"
    val docsToScore = Array(
      Map("feat_0" -> 0.04f, "feat_1" -> 0.08f, "feat_2" -> 0.01f),
      Map("feat_0" -> 0.4f, "feat_1" -> 0.8f, "feat_2" -> 0.1f)
    )
    val (query, docs) = (
      QueryContext(queryString = queryString, queryId = "1234Id"),
      docsToScore.zipWithIndex.map {
        case (map, idx) => Document(floatFeatures = map, docId = idx.toString)
      }
    )
  }
}
