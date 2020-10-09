package ml4ir.inference.tensorflow

import com.google.common.collect.ImmutableMap

import scala.collection.JavaConverters._
import ml4ir.inference.tensorflow.data.{
  ModelFeaturesConfig,
  StringMapExampleBuilder,
  StringMapSequenceExampleBuilder,
  TestData
}
import org.junit.Test
import org.junit.Assert._
import org.tensorflow.example._

@Test
class TensorFlowInferenceTest extends TestData {
  val classLoader = getClass.getClassLoader

  def resourceFor(path: String) = classLoader.getResource(path).getPath

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
  def testRankingSavedModelBundle(): Unit = {
    val bundlePath = resourceFor("ranking/model_bundle_0_0_2")
    val bundleExecutor = new SequenceExampleExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_tfrecord_sequence_example_protos",
        scoresNodeName = "StatefulPartitionedCall"
      )
    )
    val configPath = resourceFor("ranking/model_features_0_0_2.yaml")
    val modelFeatures = ModelFeaturesConfig.load(configPath)

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
  def testClassificationSavedModelBundle(): Unit = {
    val bundlePath = resourceFor("classification/simple_classification_model/tfrecord")
    val bundleExecutor = new ExampleExecutor(
      bundlePath,
      ModelExecutorConfig(
        queryNodeName = "serving_tfrecord_protos",
        scoresNodeName = "StatefulPartitionedCall_3"
      )
    )
    val configPath = resourceFor("classification/feature_config.yaml")
    val modelFeatures = ModelFeaturesConfig.load(configPath)

    val protoBuilder = StringMapExampleBuilder.withFeatureProcessors(modelFeatures,
                                                                     ImmutableMap.of(),
                                                                     ImmutableMap.of(),
                                                                     ImmutableMap.of())

    val queryContext = Map(
      "query_text" -> "king carefully there most hour long at",
      "domain_id" -> "O",
      "user_context" -> "HHH,AAA,HHH,CCC,BBB,GGG,BBB,BBB,BBB,EEE,BBB,GGG,AAA,DDD,GGG,FFF,AAA,FFF,FFF"
    )
    val proto: Example = protoBuilder.apply(queryContext.asJava)
    val predictions = bundleExecutor.apply(proto)
    /*
    trying to guess what we should have from model_predictions.csv:
b'query_id_10',"(<tf.Tensor: id=186337, shape=(), dtype=float32, numpy=0.0>, <tf.Tensor: id=186341, shape=(), dtype=float32, numpy=0.0>, <tf.Tensor: id=186345, shape=(), dtype=float32, numpy=0.0>, <tf.Tensor: id=186349, shape=(), dtype=float32, numpy=1.0>, <tf.Tensor: id=186353, shape=(), dtype=float32, numpy=0.0>, <tf.Tensor: id=186357, shape=(), dtype=float32, numpy=0.0>, <tf.Tensor: id=186361, shape=(), dtype=float32, numpy=0.0>, <tf.Tensor: id=186365, shape=(), dtype=float32, numpy=0.0>, <tf.Tensor: id=186369, shape=(), dtype=float32, numpy=0.0>)",b'king carefully there most hour long at',"(<tf.Tensor: id=188057, shape=(), dtype=string, numpy=b'king'>, <tf.Tensor: id=188061, shape=(), dtype=string, numpy=b'carefully'>, <tf.Tensor: id=188065, shape=(), dtype=string, numpy=b'there'>, <tf.Tensor: id=188069, shape=(), dtype=string, numpy=b'most'>, <tf.Tensor: id=188073, shape=(), dtype=string, numpy=b'hour'>, <tf.Tensor: id=188077, shape=(), dtype=string, numpy=b'long'>, <tf.Tensor: id=188081, shape=(), dtype=string, numpy=b'at'>, <tf.Tensor: id=188085, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188089, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188093, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188097, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188101, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188105, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188109, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188113, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188117, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188121, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188125, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188129, shape=(), dtype=string, numpy=b''>, <tf.Tensor: id=188133, shape=(), dtype=string, numpy=b''>)",b'O',"(<tf.Tensor: id=190745, shape=(), dtype=string, numpy=b'HHH'>, <tf.Tensor: id=190749, shape=(), dtype=string, numpy=b'AAA'>, <tf.Tensor: id=190753, shape=(), dtype=string, numpy=b'HHH'>, <tf.Tensor: id=190757, shape=(), dtype=string, numpy=b'CCC'>, <tf.Tensor: id=190761, shape=(), dtype=string, numpy=b'BBB'>, <tf.Tensor: id=190765, shape=(), dtype=string, numpy=b'GGG'>, <tf.Tensor: id=190769, shape=(), dtype=string, numpy=b'BBB'>, <tf.Tensor: id=190773, shape=(), dtype=string, numpy=b'BBB'>, <tf.Tensor: id=190777, shape=(), dtype=string, numpy=b'BBB'>, <tf.Tensor: id=190781, shape=(), dtype=string, numpy=b'EEE'>, <tf.Tensor: id=190785, shape=(), dtype=string, numpy=b'BBB'>, <tf.Tensor: id=190789, shape=(), dtype=string, numpy=b'GGG'>, <tf.Tensor: id=190793, shape=(), dtype=string, numpy=b'AAA'>, <tf.Tensor: id=190797, shape=(), dtype=string, numpy=b'DDD'>, <tf.Tensor: id=190801, shape=(), dtype=string, numpy=b'GGG'>, <tf.Tensor: id=190805, shape=(), dtype=string, numpy=b'FFF'>, <tf.Tensor: id=190809, shape=(), dtype=string, numpy=b'AAA'>, <tf.Tensor: id=190813, shape=(), dtype=string, numpy=b'FFF'>, <tf.Tensor: id=190817, shape=(), dtype=string, numpy=b'FFF'>, <tf.Tensor: id=190821, shape=(), dtype=string, numpy=b''>)","(<tf.Tensor: id=192993, shape=(), dtype=float32, numpy=0.07666395>, <tf.Tensor: id=192997, shape=(), dtype=float32, numpy=0.27730995>, <tf.Tensor: id=193001, shape=(), dtype=float32, numpy=0.086106114>, <tf.Tensor: id=193005, shape=(), dtype=float32, numpy=0.24008766>, <tf.Tensor: id=193009, shape=(), dtype=float32, numpy=0.3194756>, <tf.Tensor: id=193013, shape=(), dtype=float32, numpy=3.7588892e-05>, <tf.Tensor: id=193017, shape=(), dtype=float32, numpy=0.000106836895>, <tf.Tensor: id=193021, shape=(), dtype=float32, numpy=8.158483e-05>, <tf.Tensor: id=193025, shape=(), dtype=float32, numpy=0.0001307456>)"
     */
    val expected = Array(0.07666395f, 0.27730995f, 0.086106114f, 0.000106836895f, 8.158483e-05f, 0.0001307456f)
    //assertArrayEquals(expected, predictions, 1e-4f)
    System.out.println("expected: " + expected.mkString(", "))
    System.out.println("predictions: " + predictions.mkString(", "))

  }
}
