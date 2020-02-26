package ml4ir.inference.tensorflow.utils

import java.io.InputStream

import org.tensorflow.{Graph, SavedModelBundle, Session}
import org.tensorflow.example.SequenceExample
import org.tensorflow.op.io.TfRecordReader

object ModelIO {

  def loadInputStream(filePath: String): InputStream = ???

  def loadTensorflowGraph(filePath: String): Graph =
    loadTensorflowGraph(loadInputStream(filePath))

  def loadTensorflowGraph(inputStream: InputStream): Graph = {
    try {
      val bytes = Stream
        .continually(inputStream.read)
        .takeWhile(_ != -1)
        .map(_.toByte)
        .toArray
      inputStream.close()
      val graph = new Graph()
      graph.importGraphDef(bytes)
      graph
    } finally {
      inputStream.close()
    }
  }

  /**
    *
    * @param inputStream of the serialized protobuf form of the Tensorflow graph
    * @return Tensorflow Session loaded from the serialized
    */
  def loadTensorflowSession(inputStream: InputStream): Session = {
    val graph = loadTensorflowGraph(inputStream)
    new Session(graph)
  }
  /*
  def loadTensorflowSavedModelBundle(path: String): SavedModelBundle = {
    val savedModelBundle = SavedModelBundle.loader(path).load()
    savedModelBundle
  }

 */

}
