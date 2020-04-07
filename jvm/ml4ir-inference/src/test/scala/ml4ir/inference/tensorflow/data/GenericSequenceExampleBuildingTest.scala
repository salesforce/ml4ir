package ml4ir.inference.tensorflow.data

import ml4ir.inference.tensorflow.data.FeaturesConfigHelper._
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example.{Feature, FeatureList, SequenceExample}
import org.tensorflow.DataType

import scala.collection.JavaConverters._

case class SimpleQueryContext(q: String, uid: Float)

case class SimpleDocument(title: String, numClicks: Long)

@Test
class GenericSequenceExampleBuildingTest {

  @Test
  def testCaseClassSeqExampleBuilding() = {
    val modelFeatures = ModelFeaturesConfig.load(getClass.getClassLoader.getResource("model_features.yaml").getPath)

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

    val docViewsFeature: Feature = featureListMap("doc_views").getFeature(0)
    val docViewsArray: Array[Long] = docViewsFeature.getInt64List.getValueList.asScala.toList.toArray.map(_.longValue())
    assertArrayEquals("incorrectly processed document view features", Array(4L, 0L), docViewsArray)

    val titleFeature = featureListMap("doc_title").getFeature(0)
    val titleArray: Array[String] = titleFeature.getBytesList.getValueList.asScala.toArray.map(_.toStringUtf8)
    Array("the title", "yay!").zip(titleArray).foreach {
      case (title, expectedTitle) => assertEquals("incorrectly processed title", title, expectedTitle)
    }
  }
}
