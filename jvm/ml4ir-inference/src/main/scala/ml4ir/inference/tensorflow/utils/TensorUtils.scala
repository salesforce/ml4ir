package ml4ir.inference.tensorflow.utils

import org.tensorflow.Tensor

/**
  * Simple helpers for creating TensorFlow {@see Tensor}s of the right shapes.
  */
object TensorUtils {
  def replicate(encoding: Array[Float], length: Int): Array[Array[Float]] =
    Array.ofDim[Float](length, encoding.length).map(_ => encoding.clone())

  def create2Tensor(encoding: Array[Array[Float]]): Tensor[Float] =
    Tensor.create(encoding, classOf[Float])
}
