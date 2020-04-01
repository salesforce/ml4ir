package ml4ir.inference.tensorflow

import org.tensorflow.DataType

import java.util.function.{Function => JFunction}
import java.util.{Map => JMap}

package object data {
  type ServingNodeMapping = Map[String, NodeWithDefault]
  type FeaturesConfig = Map[DataType, ServingNodeMapping]
  type FnMap[T] = JMap[String, JFunction[_ >: T, _ <: T]]
}
