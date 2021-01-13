package ml4ir.inference.tensorflow

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.{ModelFeaturesConfig, StringMapExampleBuilder, TestData}
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
  class PredictionVector(var sequence: Map[String, String], var scores: Array[Float])

  /**
   * Ugly parsing of the Python CSV predicition.
   *
   * Mainly removing the "b'" formatting and slicing into List and Map.
   *
   * @param line
   * @return
   */
  def extractColumnValues(line: String): PredictionVector = {
    val cols = line.split(",").map(_.trim)

    val sequence = Map("query_text" -> cols(1).substring(2, cols(1).length - 1),
      "domain_id" -> cols(3).substring(2, cols(3).length - 1),
      "user_context" -> cols(4).substring(2, cols(4).length - 2).filterNot("'b".toSet).trim().replace(" ", ","))

    try {
      val scores = cols(7).substring(2, cols(7).length - 2).split(" ").map(_.trim).map((s: String) => s.toFloat)
      new PredictionVector(sequence, scores)
    } catch {
      case _: NumberFormatException => null
    }

  }

  /**
   * Read a prediction file from the python project.
   * The prediction csv is not 1 row per line, so we need to reconcile the file.
   *
   * @param csvFile path to the CSV
   * @return List of PredictionVector
   */
  def readPredictionCSV(csvFile: String): List[PredictionVector] = {
    val lines = Source.fromFile(csvFile).getLines.toList

    var vectorList = new ListBuffer[PredictionVector]
    var vector = ""
    for (line <- lines.drop(1)) {
      if (!line.startsWith(" ")) {
        if (vector != "") {
          vectorList ++= Option(extractColumnValues(vector))  // Don't add element if it's null
        }
        vector = line
      } else {
        vector += line
      }
    }
    vectorList.toList
  }


  @Test
  def testClassificationGeneratedModelBundle(): Unit = {
    // TODO: This need to be read from the mvn config
    val generatedBundleLocation = "../../python/"
    val bundlePath = generatedBundleLocation + "models/end_to_end_classif/final/tfrecord"
    val predictionPath = generatedBundleLocation + "logs/end_to_end_classif/model_predictions.csv"
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

