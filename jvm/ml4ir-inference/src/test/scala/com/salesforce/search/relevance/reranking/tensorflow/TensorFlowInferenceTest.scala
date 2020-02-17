package com.salesforce.search.relevance.reranking.tensorflow

import com.google.protobuf.ByteString
import ml4ir.inference.tensorflow.utils.ModelIO
import ml4ir.inference.tensorflow.{
  Document,
  PointwiseML4IRModelExecutor,
  PointwiseML4IRModelExecutorConfig,
  Query
}
import org.junit.Assert._
import org.junit._
import org.tensorflow.example._

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
    val query = "magic"
    val docsToScore = Array(
      Map("feat_0" -> 0.04f, "feat_1" -> 0.08f, "feat_2" -> 0.01f),
      Map("feat_0" -> 0.4f, "feat_1" -> 0.8f, "feat_2" -> 0.1f)
    )
    val scores = tfRecordExecutor(
      Query(queryString = query, queryId = "1234Id"),
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
  def loadTFRecord = {
    val sequenceExample: SequenceExample = SequenceExample
      .newBuilder()
      .setContext(
        Features
          .newBuilder()
          .putFeature(
            "query_txt",
            Feature
              .newBuilder()
              .setBytesList(
                BytesList
                  .newBuilder()
                  .addValue(ByteString.copyFrom("stuff".getBytes("UTF-8")))
                  .build()
              )
              .build()
          )
          .build()
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
