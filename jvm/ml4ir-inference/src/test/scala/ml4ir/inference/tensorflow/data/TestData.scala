package ml4ir.inference.tensorflow.data

import org.tensorflow.DataType

import scala.util.Random

trait TestData {
  val Q = "q"
  val UID = "userId"

  def contextFeaturesConfig: FeaturesConfig = ???
  def sequenceFeaturesConfig: FeaturesConfig = ???

  def sampleQueryContexts = {
    val stringContextConfig: Map[String, NodeWithDefault] = contextFeaturesConfig(DataType.STRING)
    val q: NodeWithDefault = stringContextConfig(Q)
    val uid: NodeWithDefault = stringContextConfig(UID)

    List(
      Map(Q -> "example query", UID -> "john.smith@example.com"),
      Map(Q -> "another" /* no UID supplied */ ),
      Map( /* no query?!? */ UID -> "user1234")
    )
  }

  def sampleDocumentExamples = {
    val stringSequenceConfig = contextFeaturesConfig(DataType.STRING)

    stringSequenceConfig.map { case (servingName, NodeWithDefault(_, defaultValue)) => }

    val floatSequenceConfig = contextFeaturesConfig(DataType.FLOAT)

  }

  def generateRandomStrings(maxStringLength: Int, num: Int, seed: Long) = {
    val start = 32
    val end = 126
    val asciiChars: Array[Char] = (start to end).map(i => i.toChar).toArray

    val rand = new Random(seed)

    for {
      i <- 0 until maxStringLength
      j <- 0 until num
    } yield {}
    val randomString =
      (0 until rand.nextInt(maxStringLength)).map(_ => asciiChars(rand.nextInt(asciiChars.length))).mkString("")

  }
  def generateRandomLongs(num: Int, seed: Long) = {}
  def generateRandomFloats(num: Int, seed: Long) = {}
}
