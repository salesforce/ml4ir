package ml4ir.inference.tensorflow.data

import java.io.File

import com.fasterxml.jackson.annotation.{JsonIgnoreProperties, JsonProperty}
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.google.common.base.Charsets._
import com.google.common.io.Files
import org.tensorflow.DataType

import scala.collection.JavaConverters._

object ModelFeaturesConfig {
  def load(configPath: String): ModelFeaturesConfig = {
    val objectMapper = new ObjectMapper(new YAMLFactory()) with ScalaObjectMapper
    objectMapper.registerModule(DefaultScalaModule)
    objectMapper.readValue[ModelFeaturesConfig](Files.readLines(new File(configPath), UTF_8).asScala.mkString("\n"))
  }
}

@JsonIgnoreProperties(ignoreUnknown = true)
case class ModelFeaturesConfig(@JsonProperty("rank") initialRank: FeatureConfig,
                               @JsonProperty("features") features: List[FeatureConfig])

@JsonIgnoreProperties(ignoreUnknown = true)
case class FeatureConfig(@JsonProperty("node_name") nodeName: String,
                         @JsonProperty("dtype") dTypeString: String,
                         @JsonProperty("serving_info") servingConfig: ServingConfig,
                         @JsonProperty("default_value") defaultValue: String,
                         @JsonProperty("tfrecord_type") tfRecordType: String) {
  def dType: DataType = DataType.valueOf(dTypeString.toUpperCase)
}

@JsonIgnoreProperties(ignoreUnknown = true)
case class ServingConfig(@JsonProperty("name") servingName: String)
