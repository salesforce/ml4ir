package ml4ir.inference.tensorflow

import ml4ir.inference.tensorflow.utils.TensorUtils
import org.tensorflow.{SavedModelBundle, Tensor, Tensors}
import com.google.protobuf.util.JsonFormat

class SavedModelBundleExecutor(dirPath: String,
                               config: PointwiseML4IRModelExecutorConfig)
    extends ((Query, Array[Document]) => Array[Float]) {
  val savedModelBundle = SavedModelBundle.load(dirPath, "serve")
  val session = savedModelBundle.session()

  override def apply(query: Query, documents: Array[Document]): Array[Float] = {
    val proto = TensorUtils.buildIRSequenceExample(
      query,
      documents,
      config.numDocsPerQuery
    )
    val inputTensor: Tensor[String] = Tensors.create(Array(proto.toByteArray))
    try {
      val ranking = Array.ofDim[Float](1, config.numDocsPerQuery)
      session
        .runner()
        .feed("serving_tfrecord_sequence_example_protos", inputTensor)
        .fetch("StatefulPartitionedCall")
        .run()
        .get(0)
        .copyTo(ranking)
      ranking(0)
    } finally {
      inputTensor.close()
    }
  }
}
