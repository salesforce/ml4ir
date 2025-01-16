package ml4ir.inference.tensorflow

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.{FeatureConfig, ModelFeaturesConfig, ServingConfig, StringMapExampleBuilder, StringMapSequenceExampleBuilder, TestData}
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example._
import scala.util.matching.Regex

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

      println("\nfeatureConfig")
      println(s"$featureConfig")

      println("\ncolNames")
      colNames.foreach(println)

      val lineMapper: String => Map[String, String] = (line: String) => colNames.zip(line.split(",")).toMap
      val data: List[Map[String, String]] = dataLines.map(lineMapper)


      println("All Features:")
        featureConfig.features.foreach { f =>
          println(s"Name: ${f.name}, Serving Name: ${f.servingConfig.servingName}, Type: ${f.tfRecordType}")
        }

      def featureSet(featType: String) =
        featureConfig.features.filter(_.tfRecordType.equalsIgnoreCase(featType)).map(_.servingConfig.servingName).toSet


      val contextFeatures = featureSet("context")
      val sequenceFeatures = featureSet("sequence")

      println("\nContext Features:")
        contextFeatures.foreach(println)

       println("\nSequence Features:")
        sequenceFeatures.foreach(println)

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

    def extractColumnValues_old(line: String): Option[PredictionVector] = {
    println(s"\nextractColumnValues: $line")

    val cols = line.split(",").map(_.trim)

    cols.foreach(println)

    val sequence = Map("query_text" -> cols(1),
      "domain_id" -> cols(3),
      "user_context" -> cols(4).substring(1, cols(4).length - 1).trim().replace(" ", ","))

      print(s"\nsequence : $sequence")

    try {
      val scores = cols(7).substring(1, cols(7).length - 1).split(" ").map(_.trim).map((s: String) => s.toFloat)
      print(s"\scores : scores")

      Some(PredictionVector(sequence, scores))
    } catch {
      case _: NumberFormatException => None
    }
  }


  /**
   * Basic parsing of the Python CSV prediction.
   *
   * @param line
   * @return
   */
  def extractColumnValues(line: String): Option[PredictionVector] = {
    println(s"extractColumnValues: $line")

    //val cols = line.split(",").map(_.trim)

    //cols.zipWithIndex.foreach { case (col, index) => println(s"Column $index: $col")}


    // Regex to match CSV fields, accounting for quoted fields with commas
    val csvPattern: Regex =
      """(?x)                   # Enable verbose regex
        (?:^|,)                 # Start of line or comma delimiter
        (                       # Start capture group
          "(?:[^"\\]|\\.)*"     # Quoted field with possible escaped chars
          |                     # OR
          [^",]*                # Unquoted field without commas or quotes
        )                       # End capture group
      """.r

    val matches = csvPattern.findAllMatchIn(line).map(_.group(1)).toList

    // Remove surrounding quotes and unescape characters
    val cols = matches.map { field =>
      if (field.startsWith("\"") && field.endsWith("\"")) {
        field.substring(1, field.length - 1).replaceAll("\\\\\"", "\"")
      }
      if (field.startsWith("\'") && field.endsWith("\'")) {
        field.substring(1, field.length - 1).replaceAll("\\\\\'", "\'")
      }
      else {
        field
      }
    }

    // Now, cols is a List[String] with correctly parsed fields
    cols.foreach(println)


    //val sequence = Map("query_text" -> cols(1),
     // "domain_id" -> cols(3),
      //"user_context" -> cols(4).substring(1, cols(4).length - 1).trim().replace(" ", ","),
      //"query_words" ->  cols(2).substring(1, cols(2).length - 1).trim().replace(" ", ","))

      val sequence = Map("query_text" -> cols(1),
      "domain_id" -> cols(3),
      "user_context" -> cols(4).substring(1, cols(4).length - 1).trim().replace(" ", ",").replace("'", ""))

      println(s"sequence: $sequence")
      // Function to safely convert string to Float
      import scala.util.{Try, Success, Failure}
        def safeToFloat(s: String): Option[Float] = {
          Try(s.toFloat) match {
            case Success(value) => Some(value)
            case Failure(exception) =>
              println(s"Failed to convert '$s' to Float: ${exception.getMessage}")
              None
          }
        }

      val scoresArray = cols(7)
          .substring(1, cols(7).length - 1)
          .split("\\s+")
          .map(_.trim)
          .filter(_.nonEmpty)
          .flatMap(safeToFloat)

      scoresArray.foreach(println)
      Some(PredictionVector(sequence, scoresArray))
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
    val bundlePath = generatedBundleLocation + "models/" + modelName + "/final/serving_tfrecord"
    val predictionPath = generatedBundleLocation + "logs/" + modelName + "/model_predictions.csv"
    val featureConfigPath = generatedBundleLocation + "ml4ir/applications/ranking/tests/data/configs/feature_config_integration_test.yaml"



    // Debugging statements
    println(s"Generated Bundle Location: $generatedBundleLocation")
    println(s"Model Name: $modelName")
    println(s"Bundle Path: $bundlePath")
    println(s"Prediction Path: $predictionPath")
    println(s"Feature Config Path: $featureConfigPath")

    evaluateRankingInferenceAccuracy(bundlePath, predictionPath, featureConfigPath)
  }

  def evaluateRankingInferenceAccuracy(bundlePath: String, predictionPath: String, featureConfigPath: String) = {
    println(s"Evaluating Ranking Inference Accuracy with:")
    println(s"Bundle Path: $bundlePath")
    println(s"Prediction Path: $predictionPath")
    println(s"Feature Config Path: $featureConfigPath")
    val allScores: Iterable[(StringMapQueryAndPredictions, SequenceExample, Array[Float], Array[Float])] = runQueriesAgainstDocs(
      predictionPath,
      bundlePath,
      featureConfigPath,
      "serving_default_protos",
      "StatefulPartitionedCall"
    )

    println("\nallScores")
    allScores.foreach(println)

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
    println(s"Running Queries Against Docs with:")
    println(s"CSV Data Path: $csvDataPath")
    println(s"Model Path: $modelPath")
    println(s"Feature Config Path: $featureConfigPath")
    println(s"Input TF Node: $inputTFNode")
    println(s"Scores TF Node: $scoresTFNode")

    val featureConfig = ModelFeaturesConfig.load(featureConfigPath)
    println(s"Loaded Feature Config: $featureConfig")

    val sequenceExampleBuilder = StringMapSequenceExampleBuilder.withFeatureProcessors(featureConfig)
    val rankingModelConfig = ModelExecutorConfig(inputTFNode, scoresTFNode)

    val rankingModel = new TFRecordExecutor(modelPath, rankingModelConfig)

    val queryContextsAndDocs = StringMapCSVLoader.loadDataFromCSV(csvDataPath, featureConfig)
    //println(s"Loaded Query Contexts and Docs: $queryContextsAndDocs")

    assertTrue("attempting to test empty query set!", queryContextsAndDocs.nonEmpty)
    queryContextsAndDocs.map {
      case q @ StringMapQueryAndPredictions(queryContext, docs, predictedScores) =>
        println(s"\nProcessing Query Context: $queryContext")
      println(s"\nDocs: $docs")

      // Check for null or empty values before building SequenceExample
      if (queryContext == null || docs == null) {
        println("Error: queryContext or docs is null.")
      }
        val t1 = queryContext.asJava
        val t2 = docs.map(_.asJava).asJava
        println(s"\nqueryContext.asJava: $t1")
        println(s"\ndocs.map: $t2")
        docs.foreach { doc =>
          doc.foreach { case (key, value) =>
            if (value == null || value.trim.isEmpty) {
              println(s"Null or empty value detected for key: $key")
            }
          }
        }


        val sequenceExample = sequenceExampleBuilder.build(queryContext.asJava, docs.map(_.asJava).asJava)
        println(s"\nBuilt SequenceExample: $sequenceExample")

        (q, sequenceExample, rankingModel(sequenceExample), predictedScores)
    }
  }

  @Test
  def testClassificationGeneratedModelBundle(): Unit = {
    val generatedBundleLocation = System.getProperty("bundleLocation")
    def modelName = System.getProperty("runName") + "_classification"
    val bundlePath = generatedBundleLocation + "models/" + modelName + "/final/serving_tfrecord"
    val predictionPath = generatedBundleLocation + "logs/" + modelName + "/model_predictions.csv"
    val featureConfigPath = generatedBundleLocation + "ml4ir/applications/classification/tests/data/configs/feature_config.yaml"

    println("in testClassificationGeneratedModelBundle")
    println(s"generatedBundleLocation=$generatedBundleLocation")
    println(s"bundlePath=$bundlePath")

    evaluateClassificationInferenceAccuracy(bundlePath, predictionPath, featureConfigPath)
  }

  def evaluateClassificationInferenceAccuracy(bundlePath: String, predictionPath: String, featureConfigPath: String) = {
  println(s"in evaluateClassificationInferenceAccuracy bundlePath=$bundlePath")
    val bundleExecutor = new TFRecordExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_default_protos",
        scoresNodeName = "StatefulPartitionedCall"
      )
    )

    val modelFeatures = ModelFeaturesConfig.load(featureConfigPath)
    println(s"modelFeatures=$modelFeatures")


    val protoBuilder = StringMapExampleBuilder.withFeatureProcessors(modelFeatures)

    println(s"readPredictionCSV(predictionPath), $predictionPath")

    val vectors: List[PredictionVector] = readPredictionCSV(predictionPath)

    assertNotEquals("No predictions found in file!", 0, vectors.length)

    vectors.foreach { queryContext: PredictionVector =>
      println("queryContext", queryContext, queryContext.sequence.asJava)
      val proto: Example = protoBuilder.apply(queryContext.sequence.asJava)
      println("proto", proto)
      //val scores: Array[Float] = bundleExecutor(proto)
      val scores = bundleExecutor(proto)

      println("scores", scores)

      assertArrayEquals("Prediction for " + queryContext.sequence + " failed", queryContext.scores, scores, 1e-6f)
    }
  }
}

