package ml4ir.inference.tensorflow.data

import java.io.FileInputStream
import java.util.{List => JList, Map => JMap}

import ml4ir.inference.tensorflow.data.FeaturesConfigHelper._
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example.{Feature, FeatureList, SequenceExample}
import org.tensorflow.DataType

import scala.collection.JavaConverters._

case class SimpleQueryContext(q: String)

case class SimpleDocument(floatFeat0: Float, floatFeat1: Float, floatFeat2: Float)

@Test
class GenericSequenceExampleBuildingTest extends TestData {
  val classLoader = getClass.getClassLoader

  @Test
  def testCaseClassSeqExampleBuilding() = {
    val modelFeatures = ModelFeaturesConfig.load(classLoader.getResource(baseConfigFile).getPath)

    val ctxProcessor = new FeaturePreprocessor[SimpleQueryContext](
      modelFeatures.toFeaturesConfig("context"),
      floatExtractor = Map.empty
        .withDefaultValue(_ => None),
      longExtractor = Map.empty
        .withDefaultValue(_ => None),
      stringExtractor = Map("q" -> ((ctx: SimpleQueryContext) => Some(ctx.q)))
        .withDefaultValue(_ => None)
    )

    val seqProcessor = new FeaturePreprocessor[SimpleDocument](
      modelFeatures.toFeaturesConfig("sequence"),
      floatExtractor = Map(
        "floatFeat0" -> ((doc: SimpleDocument) => Some(doc.floatFeat0)),
        "floatFeat1" -> ((doc: SimpleDocument) => Some(doc.floatFeat1)),
        "floatFeat2" -> ((doc: SimpleDocument) => Some(doc.floatFeat2))
      ).withDefaultValue(_ => None),
      longExtractor = Map.empty.withDefaultValue(_ => None),
      stringExtractor = Map.empty.withDefaultValue(_ => None)
    )

    val fp = new SequenceExampleBuilder[SimpleQueryContext, SimpleDocument](ctxProcessor, seqProcessor)

    val sequenceExample: SequenceExample = fp(
      SimpleQueryContext("a query!"),
      List(SimpleDocument(0.1f, 0.2f, 0.3f), SimpleDocument(1.1f, 1.2f, 1.3f))
    )
    assertNotNull(sequenceExample)

    val featureListMap: Map[String, FeatureList] = sequenceExample.getFeatureLists.getFeatureListMap.asScala.toMap

    val feat1: Feature = featureListMap("feat_1").getFeature(0)
    val feat1Array: Array[Float] = feat1.getFloatList.getValueList.asScala.toList.toArray.map(_.floatValue())
    assertArrayEquals("incorrectly processed document view features", Array(0.2f, 1.2f), feat1Array, 0.01f)

    val contextFeatures: Map[String, Feature] = sequenceExample.getContext.getFeatureMap.asScala.toMap
    val queryFeature: Feature = contextFeatures("query_str")
    val queryArray: Array[String] = queryFeature.getBytesList.getValueList.asScala.toArray.map(_.toStringUtf8)
    Array("a query!").zip(queryArray).foreach {
      case (query, expectedQuery) => assertEquals("incorrectly processed title", query, expectedQuery)
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
      //TFRecordIO.write(sequenceExample.toByteArray, basePath + "_" + i)
      //sequenceExample.writeTo(fos)
    }
    //fos.close()
  }

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
