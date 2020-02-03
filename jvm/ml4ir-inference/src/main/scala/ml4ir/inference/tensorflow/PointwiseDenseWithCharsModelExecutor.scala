package ml4ir.inference.tensorflow

import ml4ir.inference.tensorflow.utils.TensorUtils
import org.tensorflow.Session

class PointwiseDenseWithCharsModelExecutor(session: Session) {

  val INPUT_QUERY_CHAR_NODE = "INPUT_QUERY_CHAR"
  val INPUT_DOC_CHAR_NODE = "INPUT_DOC_CHAR"
  val INPUT_DOC_NUMERIC_NODE = "INPUT_DOC_NUMERIC"
  val OUTPUT_NODE = "OUTPUT"

  def apply(query: String, docs: List[(String, Array[Float])]): Array[Float] = {
    val numDocs = docs.size
    val encodedQuery: Array[Array[Float]] =
      TensorUtils.replicate(textFeatureProcessor(query), numDocs)
    val (
      encodedDocText: Array[Array[Float]],
      encodedNumericFeatures: Array[Array[Float]]
    ) =
      docs.toArray
        .map(d => (textFeatureProcessor(d._1), d._2))
        .unzip
    val ranking = Array.ofDim[Float](numDocs, 1)
    // TODO: wrap with resource catching block:
    val (inputQueryCharTensor, inputDocCharTensor, inputDocNumericTensor) =
      (
        TensorUtils.create2Tensor(encodedQuery),
        TensorUtils.create2Tensor(encodedDocText),
        TensorUtils.create2Tensor(encodedNumericFeatures)
      )
    session
      .runner()
      .feed(INPUT_QUERY_CHAR_NODE, inputQueryCharTensor)
      .feed(INPUT_DOC_CHAR_NODE, inputDocCharTensor)
      .feed(INPUT_DOC_NUMERIC_NODE, inputDocNumericTensor)
      .fetch(OUTPUT_NODE)
      .run()
      .get(0)
      .copyTo(ranking)

    ranking(0)
  }

  // TODO: ensure this is in sync with training in pythons
  def textFeatureProcessor(str: String): Array[Float] =
    str.toCharArray.map(_.toFloat)
}
