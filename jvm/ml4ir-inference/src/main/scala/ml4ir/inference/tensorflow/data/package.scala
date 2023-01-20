package ml4ir.inference.tensorflow

import org.tensorflow.proto.framework.DataType

package object data {
  type ServingNodeMapping = Map[String, NodeWithDefault]
  type FeaturesConfig = Map[DataType, ServingNodeMapping]
}
