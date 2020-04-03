package ml4ir.inference.tensorflow

import ml4ir.inference.tensorflow.data.{FeatureConfig, ServingConfig, ModelFeaturesConfig}
import org.junit.Test
import org.junit.Assert._
@Test
class ModelFeaturesConfigParserTest {

  @Test
  def testYamlParsing() = {
    val mf: ModelFeaturesConfig =
      ModelFeaturesConfig.load(getClass.getClassLoader.getResource("model_features.yaml").getPath)
    assertEquals("incorrect feature count from config", 10, mf.features.size)
    mf.features
      .map { case FeatureConfig(train, _, ServingConfig(inference), _, _) => inference -> train }
      .foreach(println)
  }
}
