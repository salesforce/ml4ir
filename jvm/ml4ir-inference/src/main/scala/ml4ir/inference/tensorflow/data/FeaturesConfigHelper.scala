package ml4ir.inference.tensorflow.data

case class NodeWithDefault(nodeName: String, defaultValue: String)

object FeaturesConfigHelper {

  /**
    * Provides implicit conversion from simple list of {@see FeatureConfig}s into a single {@code DataType}-keyed
    * {@see FeaturesConfig} object.
    * @param modelFeaturesConfig
    */
  implicit class MFConverter(modelFeaturesConfig: ModelFeaturesConfig) {
    def toFeaturesConfig(tfRecordType: String): FeaturesConfig = {
      // see ModelFeaturesConfig for more about the "rank" pseudo-feature
      val specialFeatures = if (tfRecordType.equals("sequence")) List(modelFeaturesConfig.initialRank) else List.empty
      val features = modelFeaturesConfig.features ++ specialFeatures
      val featuresConfig: FeaturesConfig =
        features
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
