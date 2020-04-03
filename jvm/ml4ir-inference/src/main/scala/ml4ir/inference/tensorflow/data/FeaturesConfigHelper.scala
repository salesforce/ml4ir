package ml4ir.inference.tensorflow.data

case class NodeWithDefault(nodeName: String, defaultValue: String)

object FeaturesConfigHelper {
  implicit class MFConverter(mf: ModelFeaturesConfig) {
    def toFeaturesConfig(tfRecordType: String): FeaturesConfig = {
      val featuresConfig: FeaturesConfig =
        mf.features
          .filter(_.tfRecordType.equalsIgnoreCase(tfRecordType))
          .groupBy(_.dType)
          .mapValues(
            _.map {
              case FeatureConfig(nodeName, _, ServingConfig(servingName), defaultValue, _) =>
                servingName -> NodeWithDefault(nodeName, defaultValue)
            }.toMap
          )
          .withDefaultValue(Map.empty)
      featuresConfig
    }
  }
}
