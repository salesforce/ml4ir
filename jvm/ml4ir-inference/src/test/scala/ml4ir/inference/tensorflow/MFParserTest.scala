package ml4ir.inference.tensorflow

import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import ml4ir.inference.tensorflow.data.MF
import org.junit.Test

import scala.io.Source
@Test
class MFParserTest {

  @Test
  def testYamlParsing() = {
    val objectMapper = new ObjectMapper(new YAMLFactory()) with ScalaObjectMapper
    objectMapper.registerModule(DefaultScalaModule)
    val yamlPath = getClass.getClassLoader.getResource("model_features.yaml").getPath
    val yamlStr = Source.fromFile(yamlPath).getLines().mkString("\n")
    val mf: MF = objectMapper.readValue[MF](yamlStr)
    mf.features.map(i => i.getServingInfo.getName -> i.getNodeName).foreach(println)
  }
}
