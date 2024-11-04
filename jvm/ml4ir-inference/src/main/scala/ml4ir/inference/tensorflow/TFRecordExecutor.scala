package ml4ir.inference.tensorflow

import com.google.protobuf.MessageLite
import org.tensorflow.ndarray.{ByteNdArray, NdArray, NdArrays, Shape, StdArrays}
import org.tensorflow.ndarray.buffer.{ByteDataBuffer, DataBuffers}
import org.tensorflow.types.{TFloat32, TString}
import org.tensorflow.{SavedModelBundle, Tensor}

import scala.collection.JavaConverters._

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
println(s"in TFRecordExecutor $dirPath")
  private val savedModelBundle = SavedModelBundle.load(dirPath, "serve")
  println("\nsavedModelBundle:\n")
  savedModelBundle.graph().operations().asScala.foreach(op => println(op.name()))

  //val metaGraphDef = MetaGraphDef.parseFrom(savedModelBundle.metaGraphDef())
  //val signatureMap = metaGraphDef.getSignatureDefMap
  //val signatureKey = "serving_tfrecord"
  //val signatureDef = signatureMap.get(signatureKey)

  //val inputInfo = signatureDef.getInputsMap.asScala
  //val outputInfo = signatureDef.getOutputsMap.asScala



  private val session = savedModelBundle.session()

  // Initialize variables if an init op exists
    try {
      session.runner().addTarget("init_op").run() // Replace "init_op" with actual name
    } catch {
      case e: Exception => println("No init_op found or initialization failed.")
    }

  // Run the initialization operation

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
    val inputTensor: TString = TString.tensorOfBytes(NdArrays.vectorOfObjects(serializeToBytes(in)))

    try {
      val resultTensor: TFloat32 = session
        .runner()
        .feed(inputNode, inputTensor)
        .fetch(outputNode)
        .run()
        .get(0)
        .asInstanceOf[TFloat32]
      resultTensor.shape().numDimensions() match {
        case 1 => StdArrays.array1dCopyOf(resultTensor)
        case 2 => StdArrays.array2dCopyOf(resultTensor)(0)
        case 3 => StdArrays.array3dCopyOf(resultTensor)(0)(0)
        case _ => throw new IllegalArgumentException("unsupported result shape: " + resultTensor.shape())
      }
    } finally {
      inputTensor.close()
    }
  }
}
