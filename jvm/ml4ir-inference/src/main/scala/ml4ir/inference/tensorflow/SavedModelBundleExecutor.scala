package ml4ir.inference.tensorflow

import ml4ir.inference.tensorflow.utils.ProtobufUtils
import org.tensorflow.{SavedModelBundle, Tensor, Tensors}

/**
  * Primary model executor for performing inference on TensorFlow's SavedModelBundle
  * usage
  *
  * @param dirPath local filesystem path to the root of the SavedModelBundle. Directory likely looks like:
  *                model_bundle/
  *                ├── assets
  *                ├── saved_model.pb
  *                └── variables
  *                    ├── variables.data-00000-of-00001
  *                    ├── variables.data-00000-of-00002
  *                    ├── variables.data-00001-of-00002
  *                    └── variables.index
  * @param config
  * @see <a href="https://www.tensorflow.org/api_docs/java/reference/org/tensorflow/SavedModelBundle">SavedModelBundle</a>
  */
class SavedModelBundleExecutor(dirPath: String, config: ModelExecutorConfig)
    extends ((QueryContext, Array[Document]) => Array[Float]) {
  val savedModelBundle = SavedModelBundle.load(dirPath, "serve")
  val session = savedModelBundle.session()

  override def apply(context: QueryContext,
                     documents: Array[Document]): Array[Float] = {
    val ModelExecutorConfig(input, output, padTo, _) = config
    val proto = ProtobufUtils.buildIRSequenceExample(context, documents, padTo)
    val inputTensor: Tensor[String] = Tensors.create(Array(proto.toByteArray))
    try {
      val ranking = Array.ofDim[Float](1, padTo)
      session
        .runner()
        .feed(input, inputTensor)
        .fetch(output)
        .run()
        .get(0)
        .copyTo(ranking)
      ranking(0)
    } finally {
      inputTensor.close()
    }
  }
}
