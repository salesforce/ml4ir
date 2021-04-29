package ml4ir.inference.tensorflow.data

import scala.collection.JavaConverters._

import java.util.{List => JList, Map => JMap}
import java.util.function.{Function => JFunction}

import com.google.common.collect.ImmutableMap

import scala.io.Source

case class StringMapQueryAndPredictions(queryContext: JMap[String, String],
                                        docs: JList[JMap[String, String]],
                                        predictedScores: Array[Float])

object StringMapCSVLoader {

  def loadDataFromCSV(dataPath: String, featureConfig: ModelFeaturesConfig): Iterable[StringMapQueryAndPredictions] = {
    val lines = Source.fromFile(dataPath).getLines().toList
    val (header, dataLines) = (lines.head, lines.tail)
    val colNames = header.split(",").map(_.replaceAll("\"", ""))
    val lineMapper: String => Map[String, String] = (line: String) =>
      colNames.zip(line.split(",")
        .map(_.replaceAll("\"", ""))
        .map(_.replaceAll("b'", ""))
        .map(_.replaceAll("\'", ""))
      ).toMap
    val data: List[Map[String, String]] = dataLines.map(lineMapper)

    def featureSet(featType: String) =
      featureConfig.features.filter(_.tfRecordType.equalsIgnoreCase(featType)).map(_.servingConfig.servingName).toSet
    val contextFeatures = featureSet("context")
    val sequenceFeatures = featureSet("sequence")

    val groupMapper = (group: List[Map[String, String]]) => {
      val context: Map[String, String] = group.head.filterKeys(contextFeatures.contains)
      val docs: List[Map[String, String]] = group.map(_.filterKeys(sequenceFeatures.contains))
      val predictedScores: Array[Float] = group.map(_.apply("ranking_score_from_python").toFloat).toArray
      (context, docs, predictedScores)
    }

    val contextsAndDocs: Iterable[(Map[String, String], List[Map[String, String]], Array[Float])] =
      data.groupBy(_("query_id")).values.map(groupMapper)

    contextsAndDocs.map(pair => StringMapQueryAndPredictions(pair._1.asJava, pair._2.map(_.asJava).asJava, pair._3))
  }

}
