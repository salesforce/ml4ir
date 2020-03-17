package ml4ir.inference.tensorflow.data

case class MultiFeatures(floatFeatures: Map[String, Float] = Map.empty,
                         int64Features: Map[String, Long] = Map.empty,
                         stringFeatures: Map[String, String] = Map.empty,
                         docMetadata: Map[String, String] = Map.empty) {
  def clean(keyFilter: String => Boolean,
            defaultFloat: Float = 0f,
            defaultLong: Long = 0L,
            defaultString: String = "") = {
    this.copy(
      floatFeatures =
        floatFeatures.filterKeys(keyFilter).withDefaultValue(defaultFloat),
      int64Features =
        int64Features.filterKeys(keyFilter).withDefaultValue(defaultLong),
      stringFeatures =
        stringFeatures.filterKeys(keyFilter).withDefaultValue(defaultString)
    )
  }
}
