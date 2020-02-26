package ml4ir.inference.tensorflow

import java.io.InputStream
import java.lang

import com.google.protobuf.ByteString
import ml4ir.inference.tensorflow.utils.{ModelIO, TensorUtils}
import org.junit.{Ignore, Test}
import org.junit.Assert._
import org.tensorflow.example._
import org.tensorflow.SavedModelBundle

import scala.collection.JavaConverters._

@Test
class TensorFlowInferenceTest {
  val classLoader = getClass.getClassLoader

  @Test
  def testLoadTFSession = {
    val graphInputStream =
      classLoader.getResourceAsStream("pointwiseModelPointwiseLoss")
    val session = ModelIO.loadTensorflowSession(graphInputStream)
    assertNotNull(session)
  }

  def runExecutorTest(
    executor: ((QueryContext, Array[Document]) => Array[Float])
  ) = {
    val query = "magic"
    val docsToScore = Array(
      Map("feat_0" -> 0.04f, "feat_1" -> 0.08f, "feat_2" -> 0.01f),
      Map("feat_0" -> 0.4f, "feat_1" -> 0.8f, "feat_2" -> 0.1f)
    )
    val scores = executor(
      QueryContext(queryString = query, queryId = "1234Id"),
      docsToScore.zipWithIndex.map {
        case (map, idx) => Document(numericFeatures = map, docId = idx.toString)
      }
    )
    scores.foreach(
      score =>
        assertTrue("all docs should score non-negative, even masks", score > 0)
    )
    for {
      maskedScore <- scores.drop(docsToScore.length)
      docScore <- scores.take(docsToScore.length)
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
      PointwiseML4IRModelExecutorConfig(
        queryNodeName = "query_str",
        scoresNodeName = "ranking_scores/Sigmoid",
        numDocsPerQuery = 25,
        queryLenMax = 20
      )
    )
    runExecutorTest(tfRecordExecutor)
  }

  @Test
  def testSavedModelBundle() = {
    val bundlePath = classLoader.getResource("model_bundle").getPath
    val bundleExecutor = new SavedModelBundleExecutor(
      bundlePath,
      PointwiseML4IRModelExecutorConfig(
        queryNodeName = "serving_tfrecord_sequence_example_protos",
        scoresNodeName = "StatefulPartitionedCall",
        numDocsPerQuery = 25,
        queryLenMax = 20
      )
    )
    runExecutorTest(bundleExecutor)
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
        case (map, idx) => Document(numericFeatures = map, docId = idx.toString)
      }
    )

  }

  def buildContextFeatures(nodePairs: (String, String)*): Features = {
    nodePairs
      .foldLeft(Features.newBuilder()) {
        case (bldr, nodePair) =>
          bldr
            .putFeature(
              nodePair._1,
              Feature
                .newBuilder()
                .setBytesList(
                  BytesList
                    .newBuilder()
                    .addValue(
                      ByteString.copyFrom(nodePair._2.getBytes("UTF-8"))
                    )
                    .build()
                )
                .build()
            )
      }
      .build()
  }

  def buildFeatureLists(documents: Array[Document],
                        numDocsPerQuery: Int): FeatureLists = {
    TensorUtils
      .transposeDocs(documents, numDocsPerQuery)
      .foldLeft(FeatureLists.newBuilder()) {
        case (bldr, (nodeName: String, featureValues: Array[Float])) =>
          bldr.putFeatureList(nodeName, buildSingleFeatureList(featureValues))
      }
      .build()
  }

  def buildSingleFeatureList(featureValues: Array[Float]): FeatureList = {
    FeatureList
      .newBuilder()
      .addFeature(
        Feature
          .newBuilder()
          .setFloatList(
            FloatList
              .newBuilder()
              .addAllValue(
                featureValues.toList.map(java.lang.Float.valueOf).asJava
              )
              .build()
          )
          .build()
      )
      .build()
  }

  def loadTFRecord = {
    SequenceExample
      .newBuilder()
      .setContext(
        buildContextFeatures("query_text" -> "stuff", "query_key" -> "dummy")
      )
      .setFeatureLists(
        FeatureLists
          .newBuilder()
          .putFeatureList(
            "feat_0",
            FeatureList
              .newBuilder()
              .addFeature(
                Feature
                  .newBuilder()
                  .setFloatList(FloatList.newBuilder().addValue(0.1f))
              )
              .build()
          )
          .putFeatureList(
            "feat_1",
            FeatureList
              .newBuilder()
              .addFeature(
                Feature
                  .newBuilder()
                  .setFloatList(FloatList.newBuilder().addValue(0.2f))
              )
              .build()
          )
          .putFeatureList(
            "feat_2",
            FeatureList
              .newBuilder()
              .addFeature(
                Feature
                  .newBuilder()
                  .setFloatList(FloatList.newBuilder().addValue(0.3f))
              )
              .build()
          )
      )
      .build()
  }
}
