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
      // the "batch" of predictions is a length-1 array (containing the array of predictions)
      val predictions: Array[Array[Float]] =
        Array.ofDim[Float](resultTensor.shape()(0).toInt, resultTensor.shape()(1).toInt)
      resultTensor.copyTo(predictions)
      predictions(0)
    } finally {
      inputTensor.close()
    }
  }
}
