package ml4ir.inference.tensorflow

import com.google.common.collect.ImmutableMap

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.{
  ModelFeaturesConfig,
  StringMapExampleBuilder,
  StringMapSequenceExampleBuilder,
  TestData
}
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example._

@Test
class TensorFlowInferenceTest extends TestData {
  val classLoader = getClass.getClassLoader

  def resourceFor(path: String) = classLoader.getResource(path).getPath

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
  def testRankingSavedModelBundle(): Unit = {
    val bundlePath = resourceFor("ranking/model_bundle_0_0_2")
    val bundleExecutor = new SequenceExampleExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_tfrecord_sequence_example_protos",
        scoresNodeName = "StatefulPartitionedCall"
      )
    )
    val configPath = resourceFor("ranking/model_features_0_0_2.yaml")
    val modelFeatures = ModelFeaturesConfig.load(configPath)

    val protoBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(modelFeatures,
                                                                             ImmutableMap.of(),
                                                                             ImmutableMap.of(),
                                                                             ImmutableMap.of())

    sampleQueryContexts.foreach { queryContext: Map[String, String] =>
      val proto: SequenceExample = protoBuilder(queryContext.asJava, sampleDocumentExamples.map(_.asJava))
      val scores: Array[Float] = bundleExecutor(proto)
      validateScores(scores, sampleDocumentExamples.length)
    }
  }

  @Test
  def testClassificationSavedModelBundle(): Unit = {
    val bundlePath = resourceFor("classification/simple_classification_model")
    val bundleExecutor = new ExampleExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_tfrecord_protos",
        scoresNodeName = "StatefulPartitionedCall_3"
      )
    )
    val configPath = resourceFor("classification/feature_config.yaml")
    val modelFeatures = ModelFeaturesConfig.load(configPath)

    val protoBuilder = StringMapExampleBuilder.withFeatureProcessors(modelFeatures,
                                                                     ImmutableMap.of(),
                                                                     ImmutableMap.of(),
                                                                     ImmutableMap.of())

    val queryContext = Map(
      "query_text" -> "a nay act hour",
      "query_words" -> "a nay act hour",
      "domain_id" -> "G",
      "user_context" -> "BBB,FFF,HHH,HHH,CCC,HHH,DDD,FFF,EEE,CCC,BBB,CCC,AAA,HHH,BBB,FFF"
    )
    val proto: Example = protoBuilder.apply(queryContext.asJava)
    val predictions = bundleExecutor.apply(proto)
    // these magic numbers are what the python side writes to model_evaluation.csv - we should get the same on the jvm
    val expected = Array(0.1992322f, 0.22705518f, 0.19074143f, 0.18944256f, 0.18591811f, 0.0015369629f, 0.0015281563f,
      0.0027524738f, 0.0017929341f)
    assertArrayEquals(expected, predictions, 1e-6f)
  }
}
