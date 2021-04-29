package ml4ir.inference.tensorflow

import java.util

import com.google.common.collect.ImmutableMap

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.{ModelFeaturesConfig, MultiFeatures, StringMapCSVLoader, StringMapExampleBuilder, StringMapQueryContextAndDocs, StringMapSequenceExampleBuilder, TestData}
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example._
@Test
class TensorFlowInferenceTest extends TestData {
  val classLoader = getClass.getClassLoader

  def resourceFor(name: String) = classLoader.getResource(name).getPath

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
    println("\n" + query.toString + "\n")
    println("as TFRecord:")
    println("\n" + sequenceExample.toString + "\n")
    println("scores: ")
    println("\n" + scores.mkString(", "))
  }

  @Test
  def testRankingSavedModelBundle(): Unit = {
    val bundlePath = resourceFor("ranking/model_bundle_0_0_2")
    val bundleExecutor = new TFRecordExecutor(
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
      val sampleDocs: List[util.Map[String, String]] = sampleDocumentExamples.map(_.asJava)
      val proto = protoBuilder(queryContext.asJava, sampleDocumentExamples.map(_.asJava))
      val scores = bundleExecutor(proto)
      validateScores(StringMapQueryContextAndDocs(queryContext.asJava, sampleDocs.asJava), proto, scores, sampleDocumentExamples.length)
    }
  }

  @Test
  def testSavedModelBundleWithCSVData(): Unit = {
    val allScores: Iterable[(StringMapQueryContextAndDocs, SequenceExample, Array[Float])] = runQueriesAgainstDocs(
      resourceFor("ranking_happy_path/model_predictions.csv"),
      resourceFor("ranking_happy_path/ranking_model_bundle"),
      resourceFor("ranking_happy_path/feature_config.yaml"),
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

    val rankingModel = new TFRecordExecutor(modelPath, rankingModelConfig)

    val queryContextsAndDocs = StringMapCSVLoader.loadDataFromCSV(csvDataPath, featureConfig)
    assertTrue("attempting to test empty query set!", queryContextsAndDocs.nonEmpty)
    queryContextsAndDocs.map {
      case q @ StringMapQueryContextAndDocs(queryContext, docs) =>
        val sequenceExample = sequenceExampleBuilder.build(queryContext, docs)
        (q, sequenceExample, rankingModel(sequenceExample))
    }
  }

  def testClassificationSavedModelBundle(): Unit = {
    val modelPath = "classification/simple_classification_model"

    val nodes = ModelExecutorConfig(
      queryNodeName = "serving_tfrecord_protos",
      scoresNodeName = "StatefulPartitionedCall_3"
    )

    val featureConfigPath = "classification/feature_config.yaml"
    val queryContext = Map(
      "query_text" -> "a nay act hour",
      "query_words" -> "a nay act hour",
      "domain_id" -> "G",
      "user_context" -> "BBB,FFF,HHH,HHH,CCC,HHH,DDD,FFF,EEE,CCC,BBB,CCC,AAA,HHH,BBB,FFF"
    )

    val predictions: Array[Float] = predict(queryContext, modelPath, featureConfigPath, nodes)

    // these magic numbers are what the python side writes to model_evaluation.csv - we should get the same on the jvm
    val expected = Array(0.1992322f, 0.22705518f, 0.19074143f, 0.18944256f, 0.18591811f, 0.0015369629f, 0.0015281563f,
      0.0027524738f, 0.0017929341f)
    assertArrayEquals(expected, predictions, 1e-6f)

    // Now use the same model, but query_text and query_words renamed to the same serving_info name 'query_text'
    val featureConfigSameNamePath = "classification/feature_config_with_same_name.yaml"
    val queryContextSameName = Map(
      "query_text" -> "a nay act hour",
      "domain_id" -> "G",
      "user_context" -> "BBB,FFF,HHH,HHH,CCC,HHH,DDD,FFF,EEE,CCC,BBB,CCC,AAA,HHH,BBB,FFF"
    )

    val predictionsSameName: Array[Float] = predict(queryContextSameName, modelPath, featureConfigSameNamePath, nodes)

    assertArrayEquals(expected, predictionsSameName, 1e-6f)
  }

  private def predict(queryContext: Map[String, String],
                      modelPath: String,
                      featureConfigPath: String,
                      nodes: ModelExecutorConfig) = {
    val bundlePath = resourceFor(modelPath)
    val bundleExecutor = new TFRecordExecutor(
      bundlePath,
      nodes
    )
    val configPath = resourceFor(featureConfigPath)
    val modelFeatures = ModelFeaturesConfig.load(configPath)

    val protoBuilder = StringMapExampleBuilder.withFeatureProcessors(modelFeatures)
    val proto: Example = protoBuilder.apply(queryContext.asJava)
    val predictions = bundleExecutor.apply(proto)
    predictions
  }
}
