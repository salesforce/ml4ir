package ml4ir.inference.tensorflow

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.{FeatureConfig, ModelFeaturesConfig, ServingConfig, StringMapExampleBuilder, StringMapSequenceExampleBuilder, TestData}
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example._

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
   * Class holding a Ranking sequence of documents from Python prediction.
   */
  case class StringMapQueryAndPredictions(queryContext: Map[String, String],
                                          docs: List[Map[String, String]],
                                          predictedScores: Array[Float])

  /**
   *
   * @param dataPath fully qualified filesystem path to the "model_predictions.csv" file as produced by
   *                 the train_inference_evaluate mode of pipeline.py (see the README at
   *                 https://github.com/salesforce/ml4ir/tree/master/python )
   * @param featureConfig the in-memory representation of the "feature_config.yaml" that the python training
   *                      process used for training
   * @return an iterable collection over what the training code got for ranking inference results, to compare with
   *         what the JVM inference sees
   */
  object StringMapCSVLoader {

    def loadDataFromCSV(dataPath: String, featureConfig: ModelFeaturesConfig): Iterable[StringMapQueryAndPredictions] = {
      val servingNameTr = featureConfig.features.map { case FeatureConfig(train, _, ServingConfig(inference, _), _) => train -> inference }.toMap

      val lines = Source.fromFile(dataPath).getLines().toList
      val (header, dataLines) = (lines.head, lines.tail)
      val colNames = header.split(",").map( n => servingNameTr.getOrElse(n, n))
      val lineMapper: String => Map[String, String] = (line: String) => colNames.zip(line.split(",")).toMap
      val data: List[Map[String, String]] = dataLines.map(lineMapper)

      def featureSet(featType: String) =
        featureConfig.features.filter(_.tfRecordType.equalsIgnoreCase(featType)).map(_.servingConfig.servingName).toSet
      val contextFeatures = featureSet("context")
      val sequenceFeatures = featureSet("sequence")

      val groupMapper = (group: List[Map[String, String]]) => {
        val context: Map[String, String] = group.head.filterKeys(contextFeatures.contains)
        val docs: List[Map[String, String]] = group.map(_.filterKeys(sequenceFeatures.contains))
        val predictedScores: Array[Float] = group.map(_.apply("ranking_score").toFloat).toArray
        (context, docs, predictedScores)
      }

      val contextsAndDocs: Iterable[(Map[String, String], List[Map[String, String]], Array[Float])] =
        data.groupBy(_("query_id")).values.map(groupMapper)

      contextsAndDocs.map(pair => StringMapQueryAndPredictions(pair._1, pair._2, pair._3))
    }

  }

  /**
   * Basic parsing of the Python CSV prediction.
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
    Source.fromFile(csvFile).getLines.toList.tail.flatMap(extractColumnValues)
  }

  def validateRankingScores(query: StringMapQueryAndPredictions,
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
    if (query.predictedScores != null) {
      // The success threshold was set to 1e-6f, but this was too strict. So we have updated to 1e-4f
      assertArrayEquals("scores aren't close enough: ", docScores, query.predictedScores, 1e-4f)
    }

  }

  @Test
  def testRankingSavedModelBundleWithCSVData(): Unit = {
    val generatedBundleLocation = System.getProperty("bundleLocation")
    def modelName = System.getProperty("runName") + "_ranking"
    val bundlePath = generatedBundleLocation + "models/" + modelName + "/final/tfrecord"
    val predictionPath = generatedBundleLocation + "logs/" + modelName + "/model_predictions.csv"
    val featureConfigPath = generatedBundleLocation + "ml4ir/applications/ranking/tests/data/configs/feature_config_integration_test.yaml"

    evaluateRankingInferenceAccuracy(bundlePath, predictionPath, featureConfigPath)
  }

  def evaluateRankingInferenceAccuracy(bundlePath: String, predictionPath: String, featureConfigPath: String) = {
    val allScores: Iterable[(StringMapQueryAndPredictions, SequenceExample, Array[Float], Array[Float])] = runQueriesAgainstDocs(
      predictionPath,
      bundlePath,
      featureConfigPath,
      "serving_tfrecord_protos",
      "StatefulPartitionedCall_1"
    )

    allScores.foreach {
      case (query: StringMapQueryAndPredictions, sequenceExample: SequenceExample, scores: Array[Float], predictedScores: Array[Float]) =>
        validateRankingScores(query, sequenceExample, scores, scores.length)
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
                             scoresTFNode: String): Iterable[(StringMapQueryAndPredictions, SequenceExample, Array[Float], Array[Float])] = {
    val featureConfig = ModelFeaturesConfig.load(featureConfigPath)
    val sequenceExampleBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(featureConfig)
    val rankingModelConfig = ModelExecutorConfig(inputTFNode, scoresTFNode)

    val rankingModel = new TFRecordExecutor(modelPath, rankingModelConfig)

    val queryContextsAndDocs = StringMapCSVLoader.loadDataFromCSV(csvDataPath, featureConfig)
    assertTrue("attempting to test empty query set!", queryContextsAndDocs.nonEmpty)
    queryContextsAndDocs.map {
      case q @ StringMapQueryAndPredictions(queryContext, docs, predictedScores) =>
        val sequenceExample = sequenceExampleBuilder.build(queryContext.asJava, docs.map(_.asJava).asJava)
        (q, sequenceExample, rankingModel(sequenceExample), predictedScores)
    }
  }

  @Test
  def testClassificationGeneratedModelBundle(): Unit = {
    val generatedBundleLocation = System.getProperty("bundleLocation")
    def modelName = System.getProperty("runName") + "_classification"
    val bundlePath = generatedBundleLocation + "models/" + modelName + "/final/tfrecord"
    val predictionPath = generatedBundleLocation + "logs/" + modelName + "/model_predictions.csv"
    //val featureConfigPath = generatedBundleLocation + "ml4ir/applications/classification/tests/data/configs/feature_config.yaml"
    // TODO: We are using another feature_config.yaml that works for this prediction, but ideally, we must used the one that has been
    // used for traning (commented above).
    val featureConfigPath = resourceFor("classification/feature_config_with_same_name.yaml")

    evaluateClassificationInferenceAccuracy(bundlePath, predictionPath, featureConfigPath)
  }

  def evaluateClassificationInferenceAccuracy(bundlePath: String, predictionPath: String, featureConfigPath: String) = {
    val bundleExecutor = new TFRecordExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_tfrecord_protos",
        scoresNodeName = "StatefulPartitionedCall_3"
      )
    )

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

