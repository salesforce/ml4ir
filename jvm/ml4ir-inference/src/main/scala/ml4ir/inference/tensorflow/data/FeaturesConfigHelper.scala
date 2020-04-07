package ml4ir.inference.tensorflow.data

case class NodeWithDefault(nodeName: String, defaultValue: String)

object FeaturesConfigHelper {
  implicit class MFConverter(mf: ModelFeaturesConfig) {
    def toFeaturesConfig(tfRecordType: String): FeaturesConfig = {
      val features = mf.features ++ (if (tfRecordType.equals("sequence")) List(mf.initialRank) else List.empty)
      val featuresConfig: FeaturesConfig =
        features
          .filter(_.tfRecordType.equalsIgnoreCase(tfRecordType))
          .groupBy(_.dType)
          .mapValues(
            _.map {
              case FeatureConfig(nodeName, _, ServingConfig(servingName, defaultValue), _) =>
                servingName -> NodeWithDefault(nodeName, defaultValue)
            }.toMap
          )
          .withDefaultValue(Map.empty)
      featuresConfig
    }
  }
}
