package ml4ir.inference.tensorflow

import org.tensorflow.example.{Example, SequenceExample}

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
class SequenceExampleExecutor(dirPath: String, config: ModelExecutorConfig)
    extends SimpleTFRecordExecutor[SequenceExample](dirPath: String, config: ModelExecutorConfig) {

  override def serializeToBytes(in: SequenceExample): Array[Byte] = in.toByteArray
}

class ExampleExecutor(dirPath: String, config: ModelExecutorConfig)
    extends SimpleTFRecordExecutor[Example](dirPath: String, config: ModelExecutorConfig) {

  override def serializeToBytes(in: Example): Array[Byte] = in.toByteArray
}
