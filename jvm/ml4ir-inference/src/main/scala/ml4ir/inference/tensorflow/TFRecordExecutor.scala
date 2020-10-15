package ml4ir.inference.tensorflow

import com.google.protobuf.MessageLite
import org.tensorflow.{SavedModelBundle, Tensor, Tensors}

/**
  * @see <a href="https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/SavedModelBundle">SavedModelBundle</a>
  * Base class for performing the actual Tensorflow execution {@see TFRecordExecutor#apply}
  *
  * @param dirPath location on local disk of the Tensorflow { @code SavedModelBundle} to be used for inference
  *                Directory likely looks like:
  *                model_bundle/
  *                ├── assets
  *                ├── saved_model.pb
  *                └── variables
  *                    ├── variables.data-00000-of-00001
  *                    ├── variables.data-00000-of-00002
  *                    ├── variables.data-00001-of-00002
  *                    └── variables.index
  * @param config instantiated struct with the values from the feature_config.yaml relevant for inference
  */
class TFRecordExecutor(dirPath: String, config: ModelExecutorConfig) {
  private val savedModelBundle = SavedModelBundle.load(dirPath, "serve")
  private val session = savedModelBundle.session()

  /**
    * To serialize the protobuf input into a byte[] as input to the Tensorflow graph
    * @param in Should be either a { @see org.tensorflow.example.Example} or { @see org.tensorflow.SequenceExample},
    *           likely constructed by { @see ExampleBuilder} or { @see SequenceExampleBuilder} respectively
    * @return serialized bytes of the protobuf, as specified by the {@code MessageLite} interface
    */
  def serializeToBytes(in: MessageLite): Array[Byte] = in.toByteArray

  /**
    *
    * @param in
    * @return
    */
  def apply(in: MessageLite): Array[Float] = {
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
