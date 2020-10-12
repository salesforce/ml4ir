package ml4ir.inference.tensorflow

import ml4ir.inference.tensorflow.data.{FeatureConfig, ServingConfig, ModelFeaturesConfig}
import org.junit.Test
import org.junit.Assert._
@Test
class ModelFeaturesConfigParserTest {
  val classLoader = getClass.getClassLoader

  def pathFor(name: String) = classLoader.getResource("ranking/" + name).getPath
  val configFile = "model_features_0_0_2.yaml"

  @Test
  def testYamlParsing() = {
    val mf: ModelFeaturesConfig = ModelFeaturesConfig.load(pathFor(configFile))
    assertEquals("incorrect feature count from config", 5, mf.features.size)
    // TODO: actually validate some values?
    mf.features
      .map { case FeatureConfig(train, _, ServingConfig(inference, _), _) => inference -> train }
      .foreach(println)
  }
}
