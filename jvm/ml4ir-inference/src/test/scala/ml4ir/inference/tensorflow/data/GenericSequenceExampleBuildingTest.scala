package ml4ir.inference.tensorflow.data

import java.io.{FileInputStream, FileOutputStream}
import java.util.{List => JList, Map => JMap}
import java.util

import ml4ir.inference.tensorflow.data.FeaturesConfigHelper._
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example.{Feature, FeatureList, SequenceExample}
import org.tensorflow.DataType

import scala.collection.JavaConverters._

case class SimpleQueryContext(q: String, uid: Float)

case class SimpleDocument(title: String, numClicks: Long)

@Test
class GenericSequenceExampleBuildingTest extends TestData {
  val classLoader = getClass.getClassLoader

  @Test
  def testCaseClassSeqExampleBuilding() = {
    val modelFeatures = ModelFeaturesConfig.load(classLoader.getResource(baseConfigFile).getPath)

    val ctxProcessor = new FeaturePreprocessor[SimpleQueryContext](
      modelFeatures.toFeaturesConfig("context"),
      Map("uid" -> ((ctx: SimpleQueryContext) => Some(ctx.uid)), "unknown" -> ((_: SimpleQueryContext) => Some(1f)))
        .withDefaultValue(_ => None),
      Map.empty
        .withDefaultValue(_ => None),
      Map("q" -> ((ctx: SimpleQueryContext) => Some(ctx.q)))
        .withDefaultValue(_ => None)
    )

    val seqProcessor = new FeaturePreprocessor[SimpleDocument](
      modelFeatures.toFeaturesConfig("sequence"),
      Map.empty.withDefaultValue(_ => None),
      Map("numDocumentViews" -> ((doc: SimpleDocument) => Some(doc.numClicks))).withDefaultValue(_ => None),
      Map("docTitle" -> ((doc: SimpleDocument) => Some(doc.title))).withDefaultValue(_ => None),
      Map(DataType.STRING -> Map("docTitle" -> new PrimitiveProcessor() {
        override def processString(s: String): String = s.toLowerCase
      }))
    )

    val fp = new SequenceExampleBuilder[SimpleQueryContext, SimpleDocument](ctxProcessor, seqProcessor)

    val sequenceExample: SequenceExample = fp(
      SimpleQueryContext("a query", 123f),
      List(SimpleDocument("The title", 4), SimpleDocument("Yay!", 0))
    )
    assertNotNull(sequenceExample)

    val featureListMap: Map[String, FeatureList] = sequenceExample.getFeatureLists.getFeatureListMap.asScala.toMap

    val docViewsFeature: Feature = featureListMap("feat_1").getFeature(0)
    val docViewsArray: Array[Long] = docViewsFeature.getInt64List.getValueList.asScala.toList.toArray.map(_.longValue())
    assertArrayEquals("incorrectly processed document view features", Array(4L, 0L), docViewsArray)

    val titleFeature = featureListMap("doc_title").getFeature(0)
    val titleArray: Array[String] = titleFeature.getBytesList.getValueList.asScala.toArray.map(_.toStringUtf8)
    Array("the title", "yay!").zip(titleArray).foreach {
      case (title, expectedTitle) => assertEquals("incorrectly processed title", title, expectedTitle)
    }
  }

  private def getQueryContexts(numExamples: Int,
                               modelFeaturesConfig: ModelFeaturesConfig): List[JMap[String, String]] = {
    (0 until numExamples)
      .map { i =>
        Map(Q -> ("query " + i))
      }
      .toList
      .map(_.asJava)
  }

  private def getExamples(numExamples: Int, numResultsPerExample: Int, modelFeaturesConfig: ModelFeaturesConfig) = {
    case class QueryResultAndRank(queryId: Int, rank: Int)
    val flattenedFeatures: List[(QueryResultAndRank, Map[String, String])] = (for {
      exampleNum: Int <- 0 until numExamples
      rank: Int <- 0 until numResultsPerExample
      (dataType: DataType, snm: Map[String, NodeWithDefault]) <- modelFeaturesConfig.toFeaturesConfig("sequence")
      (featureName: String, NodeWithDefault(nodeName: String, defaultValue: String)) <- snm
    } yield {
      QueryResultAndRank(exampleNum, rank) -> (dataType match {
        case DataType.FLOAT  => Map(featureName -> s"${exampleNum + rank}")
        case DataType.INT64  => Map(featureName -> s"${(100 * exampleNum) + rank}")
        case DataType.STRING => Map(featureName -> s"${featureName}_${exampleNum}_$rank")
      })
    }).toList
    flattenedFeatures
      .groupBy(_._1.queryId)
      .map { // group by queryId: all different features and different ranked results
        case (_: Int, featurePairMaps: List[(QueryResultAndRank, Map[String, String])]) => {
          val queryResults: List[Map[String, String]] = featurePairMaps
            .groupBy(_._1.rank)
            .values // drop the rank grouping key
            .map(_.map(_._2)) // just: List(Map("feature1" -> "value1"), Map("feature2" -> "value2"), ...)
            .toList
            .reduce(_ ++ _) // Map("feature1" -> "value1", "feature2" -> "value2", ... )
          queryResults.map(_.asJava).asJava
        }
      }
      .toList
  }
  @Test
  @throws[Exception]
  def serializeTestData(): Unit = {
    val modelFeatures: ModelFeaturesConfig = ModelFeaturesConfig.load(classLoader.getResource(baseConfigFile).getPath)
    val sequenceExampleBuilder: SequenceExampleBuilder[JMap[String, String], JMap[String, String]] =
      StringMapSequenceExampleBuilder.withFeatureProcessors(modelFeatures)
    val numExamples: Int = 10
    val numResultsPerExample: Int = 25
    val queryContexts: List[JMap[String, String]] = getQueryContexts(numExamples, modelFeatures)
    val exampleLists: List[JList[JMap[String, String]]] = getExamples(numExamples, numResultsPerExample, modelFeatures)
    val basePath = "/tmp/tfRecord"
    //val fos: FileOutputStream = new FileOutputStream("/tmp/seqExamples.proto")
    for {
      ((queryContext, exampleList), i) <- queryContexts.zip(exampleLists).zipWithIndex
    } {
      val sequenceExample: SequenceExample = sequenceExampleBuilder.build(queryContext, exampleList)
      TFRecordIO.write(sequenceExample.toByteArray, basePath + "_" + i)
      //sequenceExample.writeTo(fos)
    }
    //fos.close()
  }

  @Test
  def loadTestData() = {
    val proto: SequenceExample = SequenceExample.parseFrom(new FileInputStream("/tmp/seqExamples.proto"))
    println(proto.toString)
    assertNotNull(proto)
  }
}

object SerializationHelper extends TestData {
  val classLoader = getClass.getClassLoader

  private def getQueryContexts(numExamples: Int,
                               modelFeaturesConfig: ModelFeaturesConfig): List[JMap[String, String]] = {
    (0 until numExamples)
      .map { i =>
        Map(Q -> ("query " + i))
      }
      .toList
      .map(_.asJava)
  }

  private def getExamples(numExamples: Int, numResultsPerExample: Int, modelFeaturesConfig: ModelFeaturesConfig) = {
    case class QueryResultAndRank(queryId: Int, rank: Int)
    val flattenedFeatures: List[(QueryResultAndRank, Map[String, String])] = (for {
      exampleNum: Int <- 0 until numExamples
      rank: Int <- 0 until numResultsPerExample
      (dataType: DataType, snm: Map[String, NodeWithDefault]) <- modelFeaturesConfig.toFeaturesConfig("sequence")
      (featureName: String, NodeWithDefault(nodeName: String, defaultValue: String)) <- snm
    } yield {
      QueryResultAndRank(exampleNum, rank) -> (dataType match {
        case DataType.FLOAT  => Map(featureName -> s"${exampleNum + rank}")
        case DataType.INT64  => Map(featureName -> s"${(100 * exampleNum) + rank}")
        case DataType.STRING => Map(featureName -> s"${featureName}_${exampleNum}_$rank")
      })
    }).toList
    flattenedFeatures
      .groupBy(_._1.queryId)
      .map { // group by queryId: all different features and different ranked results
        case (_: Int, featurePairMaps: List[(QueryResultAndRank, Map[String, String])]) => {
          val queryResults: List[Map[String, String]] = featurePairMaps
            .groupBy(_._1.rank)
            .values // drop the rank grouping key
            .map(_.map(_._2)) // just: List(Map("feature1" -> "value1"), Map("feature2" -> "value2"), ...)
            .toList
            .reduce(_ ++ _) // Map("feature1" -> "value1", "feature2" -> "value2", ... )
          queryResults.map(_.asJava).asJava
        }
      }
      .toList
  }

  def sampleSequenceExamples() = {
    val modelFeatures: ModelFeaturesConfig = ModelFeaturesConfig.load(classLoader.getResource(baseConfigFile).getPath)
    val sequenceExampleBuilder: SequenceExampleBuilder[JMap[String, String], JMap[String, String]] =
      StringMapSequenceExampleBuilder.withFeatureProcessors(modelFeatures)
    val numExamples: Int = 10
    val numResultsPerExample: Int = 25
    val queryContexts: List[JMap[String, String]] = getQueryContexts(numExamples, modelFeatures)
    val exampleLists: List[JList[JMap[String, String]]] = getExamples(numExamples, numResultsPerExample, modelFeatures)
    val basePath = "/tmp/tfRecord"
    //val fos: FileOutputStream = new FileOutputStream("/tmp/seqExamples.proto")
    for {
      ((queryContext, exampleList), i) <- queryContexts.zip(exampleLists).zipWithIndex
    } yield {
      sequenceExampleBuilder.build(queryContext, exampleList)
    }
  }
}
