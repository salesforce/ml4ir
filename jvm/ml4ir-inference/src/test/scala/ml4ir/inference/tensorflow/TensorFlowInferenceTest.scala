package ml4ir.inference.tensorflow

import java.io.InputStream

import com.google.common.collect.ImmutableMap

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.{
  Example,
  FeatureProcessors,
  ModelFeaturesConfig,
  MultiFeatures,
  StringMapCSVLoader,
  StringMapQueryContextAndDocs,
  StringMapSequenceExampleBuilder,
  TestData
}
import org.junit.{Ignore, Test}
import org.junit.Assert._
import org.tensorflow.example._
@Test
class TensorFlowInferenceTest extends TestData {
  val classLoader = getClass.getClassLoader

  def pathFor(name: String) = classLoader.getResource(name).getPath

  def validateScores(query: StringMapQueryContextAndDocs,
                     sequenceExample: SequenceExample,
                     scores: Array[Float],
                     numDocs: Int) = {
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
    println("input, as java object:")
    println("\n" + query.toString)
    println("as TFRecord:")
    println("\n" + sequenceExample.toString)
    println("scores: ")
    println("\n" + scores.mkString(", "))
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

    sampleQueryContexts.take(1).foreach { queryContext: Map[String, String] =>
      val proto = protoBuilder(queryContext.asJava, sampleDocumentExamples.map(_.asJava))
      val scores = bundleExecutor(proto)
      validateScores(null, proto, scores, sampleDocumentExamples.length)
    }
  }

  @Test
  def testSavedModelBundleWithCSVData(): Unit = {
    val allScores: Iterable[(StringMapQueryContextAndDocs, SequenceExample, Array[Float])] = runQueriesAgainstDocs(
      pathFor("file_0.csv"),
      pathFor("activate_model_bundle"),
      pathFor("activate_feature_config.yaml"),
      "serving_tfrecord_protos",
      "StatefulPartitionedCall_1"
    )

    allScores.take(1).foreach {
      case (query: StringMapQueryContextAndDocs, sequenceExample: SequenceExample, scores: Array[Float]) =>
        validateScores(query, sequenceExample, scores, scores.length)
    }
  }

  /**
    * Helper method to produce scores for a model, given input CSV-formatted test data
    *
    * @param csvDataPath path to CSV-formatted training/test data
    * @param modelPath path to SavedModelBundle
    * @param featureConfigPath path to feature_config.yaml
    * @param inputTFNode tensorflow graph node name to feed in SequencExamples for scoring
    * @param scoresTFNode tensorflow graph node name to fetch the scores
    * @return scores for each input
    */
  def runQueriesAgainstDocs(
      csvDataPath: String,
      modelPath: String,
      featureConfigPath: String,
      inputTFNode: String,
      scoresTFNode: String): Iterable[(StringMapQueryContextAndDocs, SequenceExample, Array[Float])] = {
    val featureConfig = ModelFeaturesConfig.load(featureConfigPath)
    val sequenceExampleBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(featureConfig)
    val rankingModelConfig = ModelExecutorConfig(inputTFNode, scoresTFNode)
    val rankingModel = new SavedModelBundleExecutor(modelPath, rankingModelConfig)

    val queryContextsAndDocs = StringMapCSVLoader.loadDataFromCSV(csvDataPath, featureConfig)
    assertTrue("attempting to test empty query set!", queryContextsAndDocs.nonEmpty)
    queryContextsAndDocs.map {
      case q @ StringMapQueryContextAndDocs(queryContext, docs) =>
        val sequenceExample = sequenceExampleBuilder.build(queryContext, docs)
        (q, sequenceExample, rankingModel(sequenceExample))
    }
  }

}
