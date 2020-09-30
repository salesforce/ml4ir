package ml4ir.inference.tensorflow

import org.tensorflow.{SavedModelBundle, Tensor, Tensors}

import scala.reflect.ClassTag

abstract class SimpleTFRecordExecutor[IN](dirPath: String, config: ModelExecutorConfig) {
  private val savedModelBundle = SavedModelBundle.load(dirPath, "serve")
  private val session = savedModelBundle.session()

  def serializeToBytes(in: IN): Array[Byte]

  def apply(in: IN): Array[Float] = {
    val ModelExecutorConfig(inputNode, outputNode) = config
    // TF typically runs the forward pass on batches, so we wrap our protobuf bytes in another outer length-1 array
    val inputTensor: Tensor[String] = Tensors.create(Array(serializeToBytes(in)))
    try {
      val resultTensor: Tensor[_] = session
        .runner()
        .feed(inputNode, inputTensor)
        .fetch(outputNode)
        .run()
        .get(0)
      resultTensor.shape().length match {
        case 1 =>
          val predictions: Array[Float] = Array.ofDim[Float](resultTensor.shape()(0).toInt)
          resultTensor.copyTo(predictions)
          predictions
        case 2 =>
          val predictions: Array[Array[Float]] =
            Array.ofDim[Float](resultTensor.shape()(0).toInt, resultTensor.shape()(1).toInt)
          resultTensor.copyTo(predictions)
          predictions(0)
        case 3 =>
          val predictions: Array[Array[Array[Float]]] =
            Array
              .ofDim[Float](resultTensor.shape()(0).toInt, resultTensor.shape()(1).toInt, resultTensor.shape()(2).toInt)
          resultTensor.copyTo(predictions)
          predictions(0)(0)
        case _ =>
          throw new IllegalArgumentException("unsupported result shape: " + resultTensor.shape())
      }
    } finally {
      inputTensor.close()
    }
  }
}
