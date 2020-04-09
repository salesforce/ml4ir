package ml4ir.inference.tensorflow

import java.io.InputStream

import com.google.common.collect.ImmutableMap
import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.{
  Example,
  FeatureProcessors,
  ModelFeaturesConfig,
  MultiFeatures,
  StringMapSequenceExampleBuilder,
  TestData
}
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

  @Test
  def testSavedModelBundle(): Unit = {
    val bundlePath = classLoader.getResource("model_bundle_0_0_2").getPath
    val bundleExecutor = new SavedModelBundleExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_tfrecord_sequence_example_protos",
        scoresNodeName = "StatefulPartitionedCall"
      )
    )
    val configPath = classLoader.getResource("model_features_0_0_2.yaml").getPath
    val modelFeatures = ModelFeaturesConfig.load(configPath)

    val protoBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(modelFeatures,
                                                                             ImmutableMap.of(),
                                                                             ImmutableMap.of(),
                                                                             ImmutableMap.of())

    sampleQueryContexts.foreach { queryContext: Map[String, String] =>
      val proto = protoBuilder(queryContext.asJava, sampleDocumentExamples.map(_.asJava))
      val scores = bundleExecutor(proto)
      validateScores(scores, sampleDocumentExamples.length)
    }
  }
}
