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
    val lines = Source.fromFile(csvFile).getLines.toList

    var vectorList = new ListBuffer[PredictionVector]
    for (line <- lines.drop(1)) { //skip the header
      vectorList ++= extractColumnValues(line)  // Don't add element if it's null
    }
    vectorList.toList
  }


  @Test
  def testClassificationGeneratedModelBundle(): Unit = {
    val generatedBundleLocation = System.getProperty("bundleLocation")
    def modelName = System.getProperty("runName")
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

