package ml4ir.inference.tensorflow.data

import org.tensorflow.DataType

case class MultiFeatures(floatFeatures: Map[String, Float] = Map.empty,
                         int64Features: Map[String, Long] = Map.empty,
                         stringFeatures: Map[String, String] = Map.empty,
                         docMetadata: Map[String, String] = Map.empty) {
//  def clean(keyFilter: Map[DataType, String => Boolean],
//            defaultFloat: Float = 0f,
//            defaultLong: Long = 0L,
//            defaultString: String = "") = {
//    this.copy(
//      floatFeatures = floatFeatures
//        .filterKeys(keyFilter(DataType.FLOAT))
//        .withDefaultValue(defaultFloat),
//      int64Features = int64Features
//        .filterKeys(keyFilter(DataType.INT64))
//        .withDefaultValue(defaultLong),
//      stringFeatures = stringFeatures
//        .filterKeys(keyFilter(DataType.STRING))
//        .withDefaultValue(defaultString)
//    )
//  }
}
