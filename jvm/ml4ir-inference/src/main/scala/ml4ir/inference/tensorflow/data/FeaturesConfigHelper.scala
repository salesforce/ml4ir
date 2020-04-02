package ml4ir.inference.tensorflow.data

import scala.collection.JavaConverters._
import org.tensorflow.DataType

case class NodeWithDefault(nodeName: String, defaultValue: String)

object FeaturesConfigHelper {
  implicit class ModelFeaturesConverter(modelFeatures: ModelFeatures) {
    def toFeaturesConfig(tfRecordType: String): FeaturesConfig = {
      val featuresConfig: FeaturesConfig =
        modelFeatures.getFeatures.asScala.toList
          .filter(_.getTfRecordType.equalsIgnoreCase(tfRecordType))
          .groupBy(
            inputFeature => DataType.valueOf(inputFeature.getDtype.toUpperCase)
          )
          .mapValues(
            _.map(
              feature => feature.getServingInfo.getName -> NodeWithDefault(feature.getNodeName, feature.getDefaultValue)
            ).toMap
          )
          .withDefaultValue(Map.empty)
      featuresConfig
    }
  }
}
