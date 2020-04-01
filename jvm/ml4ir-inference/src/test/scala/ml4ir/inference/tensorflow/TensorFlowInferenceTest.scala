package ml4ir.inference.tensorflow

import java.io.InputStream

import ml4ir.inference.tensorflow.data.{Example, MultiFeatures, TestData}
import org.junit.{Ignore, Test}
import org.junit.Assert._
import org.tensorflow.example._
@Test
class TensorFlowInferenceTest extends TestData {
  val classLoader = getClass.getClassLoader

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
  /*
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
    val (queryContext: Example, docs: Array[Example]) = testQueries
    val protoBuilder = SequenceExampleBuilder(
      FeatureConfig(
        contextFeatures = List(FeatureField("query_text", "query_text", DataType.STRING, "")),
        documentFeatures = List(
          FeatureField("feat_0", "feat_0", DataType.FLOAT, "0"),
          FeatureField("feat_1", "feat_1", DataType.FLOAT, "0"),
          FeatureField("feat_2", "feat_2", DataType.FLOAT, "0"),
          FeatureField("pos", "pos", DataType.INT64, "0")
        )
      )
    )
    val proto  = protoBuilder(queryContext, docs)
    val scores = bundleExecutor(proto)
    validateScores(scores, docs.length)
  }

   */

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
      Example(
        features = MultiFeatures(stringFeatures = Map("query_text" -> queryString))
      ),
      docsToScore.zipWithIndex.map {
        case (map, idx) =>
          Example(
            features = MultiFeatures(floatFeatures = map)
          )
      }
    )
  }

  def testQueries: (Example, Array[Example]) = {
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
      ),
      Map("feat_0" -> 0.8f, "fake_feat" -> -1f)
    )
    (
      Example(
        features = MultiFeatures(stringFeatures = Map("query_text" -> query))
      ),
      docsToScore.zipWithIndex.map {
        case (map, idx) =>
          Example(
            features = MultiFeatures(
              floatFeatures = map,
              int64Features = Map("pos" -> idx.toLong)
            )
          )
      }
    )
  }

}
