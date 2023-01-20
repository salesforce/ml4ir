package ml4ir.inference.tensorflow


package object data {
  type ServingNodeMapping = Map[String, NodeWithDefault]
  type FeaturesConfig = Map[DataType, ServingNodeMapping]
}
