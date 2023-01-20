package ml4ir.inference.tensorflow.data

import java.io.File
import com.fasterxml.jackson.annotation.{JsonIgnoreProperties, JsonProperty}
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory
import com.fasterxml.jackson.module.scala.experimental.ScalaObjectMapper
import com.fasterxml.jackson.module.scala.DefaultScalaModule
import com.google.common.base.Charsets._
import com.google.common.io.Files
import org.tensorflow.proto.framework.DataType
import org.tensorflow.types.{TFloat32, TInt64, TString}
import org.tensorflow.types.family.TType

import scala.collection.JavaConverters._

object ModelFeaturesConfig {
  def load(configPath: String): ModelFeaturesConfig = {
    val objectMapper = new ObjectMapper(new YAMLFactory()) with ScalaObjectMapper
    objectMapper.registerModule(DefaultScalaModule)
    objectMapper.readValue[ModelFeaturesConfig](Files.readLines(new File(configPath), UTF_8).asScala.mkString("\n"))
  }
}

/**
  * The configuration for performing model inference:
  * @param initialRank a special required top-level "feature": defines a Tensorflow node in the model which describes
  *                    which position in the candidate set was each candidate found (e.g. by an earlier round of
  *                    scoring / retrieval).  It need not be supplied at inference-time, but is required in the config.
  * @param features both a list of mappings between training features and their runtime name, as well as the datatype
  *                 and default value if not supplied.
  */
@JsonIgnoreProperties(ignoreUnknown = true)
case class ModelFeaturesConfig(@JsonProperty("rank") initialRank: FeatureConfig,
                               @JsonProperty("features") features: List[FeatureConfig])

/**
  * At inference-time, ml4ir's FeatureConfig needs to know only the following properties per feature:
  * @param name the protobuf key name representing the input. Tensorflow will wire the value into the corresponding node
  *             name
  * @param dTypeString string | int64 | float
  * @param servingConfig {@see ServingConfig} below
  * @param tfRecordType context | sequence (latter only for {@see SequenceExample} features)
  */
@JsonIgnoreProperties(ignoreUnknown = true)
case class FeatureConfig(@JsonProperty("name") name: String,
                         @JsonProperty("dtype") dTypeString: String,
                         @JsonProperty("serving_info") servingConfig: ServingConfig,
                         @JsonProperty("tfrecord_type") tfRecordType: String) {
  def dType: DataType = {
    dTypeString.toUpperCase match {
      case "INT" => DataType.DT_INT64
      case "INT64" => DataType.DT_INT64
      case "FLOAT" => DataType.DT_FLOAT
      case "FLOAT32" => DataType.DT_FLOAT
      case "BYTES" => DataType.DT_STRING
      case "STRING" => DataType.DT_STRING
    }
  }
}

/**
  * {@see FeatureConfig} parameters only used / relevant at serving/inference time - can be ignored during training, and
  * in fact added onto the config later or changed at any time, without retraining
  * @param servingName The key to look up the feature's value in an e.g. input {@code HashMap}.  This name is not
  *      referenced in the serialized model, instead this config specifies the serving time -> training time mapping of
  *      serving_info.name -> node_name
  * @param defaultValue if this feature is not present in the input, a String form of this value will be supplied to
  *                     the TF model.
  */
@JsonIgnoreProperties(ignoreUnknown = true)
case class ServingConfig(@JsonProperty("name") servingName: String, @JsonProperty("default_value") defaultValue: String)
