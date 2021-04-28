package ml4ir.inference.tensorflow

import com.google.common.collect.ImmutableMap

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.{ModelFeaturesConfig, StringMapExampleBuilder, StringMapSequenceExampleBuilder, TestData}
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example._

import scala.collection.mutable.ListBuffer
import scala.io.Source

@Test
class TensorFlowInferenceIT extends TestData {
  val classLoader = getClass.getClassLoader

  def resourceFor(path: String) = classLoader.getResource(path).getPath

  /**
   * Class holding a Example and prediction from Python
   * @param sequence the Example context
   * @param scores the prediction scores
   */
  case class PredictionVector(sequence: Map[String, String], scores: Array[Float])

  /**
   * Basic parsing of the Python CSV predicition.
   *
   * @param line
   * @return
   */
  def extractColumnValues(line: String): Option[PredictionVector] = {
    val cols = line.split(",").map(_.trim)

    val sequence = Map("query_text" -> cols(1),
      "domain_id" -> cols(3),
      "user_context" -> cols(4).substring(1, cols(4).length - 1).trim().replace(" ", ","))

    try {
      val scores = cols(7).substring(1, cols(7).length - 1).split(" ").map(_.trim).map((s: String) => s.toFloat)
      Some(PredictionVector(sequence, scores))
    } catch {
      case _: NumberFormatException => None
    }
  }

  /**
   * Read a prediction file from the python project.
   *
   * @param csvFile path to the CSV
   * @return List of PredictionVector
   */
  def readPredictionCSV(csvFile: String): List[PredictionVector] = {
    return Source.fromFile(csvFile).getLines.toList.tail.flatMap(extractColumnValues)
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
  def testRankingGeneratedModelBundle(): Unit = {
    val generatedBundleLocation = "/Users/aducoulombier/projects/ml4ir_back/python/" //System.getProperty("bundleLocation")
    def modelName = "alain_ranking" //System.getProperty("runName") + "_classification"
    val bundlePath = generatedBundleLocation + "models/" + modelName + "/final/tfrecord"
    val predictionPath = generatedBundleLocation + "logs/" + modelName + "/model_predictions.csv"
    //val featureConfigPath = generatedBundleLocation + "ml4ir/applications/classification/tests/data/configs/feature_config.yaml"

    val bundleExecutor = new TFRecordExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_tfrecord_protos",
        scoresNodeName = "StatefulPartitionedCall_1"
      )
    )

    // TODO: using the model yaml don't work.
    val featureConfigPath = "/Users/aducoulombier/projects/ml4ir_back/python/ml4ir/applications/ranking/tests/data/config/feature_config.yaml"
    val modelFeatures = ModelFeaturesConfig.load(featureConfigPath)


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
  def testClassificationGeneratedModelBundle(): Unit = {
    val generatedBundleLocation = "../../python/" //System.getProperty("bundleLocation")
    def modelName = "end_to_end_test_classification" //System.getProperty("runName") + "_classification"
    val bundlePath = generatedBundleLocation + "models/" + modelName + "/final/tfrecord"
    val predictionPath = generatedBundleLocation + "logs/" + modelName + "/model_predictions.csv"
    //val featureConfigPath = generatedBundleLocation + "ml4ir/applications/classification/tests/data/configs/feature_config.yaml"

    val bundleExecutor = new TFRecordExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_tfrecord_protos",
        scoresNodeName = "StatefulPartitionedCall_3"
      )
    )

    // TODO: using the model yaml don't work.
    val featureConfigPath = resourceFor("classification/feature_config_with_same_name.yaml")
    val modelFeatures = ModelFeaturesConfig.load(featureConfigPath)


    val protoBuilder = StringMapExampleBuilder.withFeatureProcessors(modelFeatures)

    val vectors: List[PredictionVector] = readPredictionCSV(predictionPath)

    assertNotEquals("No predictions found in file!", 0, vectors.length)

    vectors.foreach { queryContext: PredictionVector =>
      val proto: Example = protoBuilder.apply(queryContext.sequence.asJava)
      val scores: Array[Float] = bundleExecutor(proto)
      assertArrayEquals("Prediction for " + queryContext.sequence + " failed", queryContext.scores, scores, 1e-6f)
    }
  }
}

