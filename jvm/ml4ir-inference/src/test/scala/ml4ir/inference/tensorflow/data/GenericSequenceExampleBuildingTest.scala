package ml4ir.inference.tensorflow.data

import ml4ir.inference.tensorflow.data.FeaturesConfigHelper._
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example.{Feature, FeatureList, SequenceExample}

import scala.collection.JavaConverters._

case class SimpleQueryContext(q: String)

case class SimpleDocument(floatFeat0: Float, floatFeat1: Float, floatFeat2: Float)

@Test
class GenericSequenceExampleBuildingTest extends TestData {
  val classLoader = getClass.getClassLoader

  def pathFor(name: String) = classLoader.getResource("ranking/" + name).getPath

  @Test
  def testCaseClassSeqExampleBuilding() = {
    val modelFeatures = ModelFeaturesConfig.load(pathFor(baseConfigFile))

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
}
