package ml4ir.inference.tensorflow.utils

import com.google.protobuf.ByteString
import ml4ir.inference.tensorflow.{Document, QueryContext}
import org.tensorflow.Tensor
import org.tensorflow.example.{
  BytesList,
  Feature,
  FeatureList,
  FeatureLists,
  Features,
  FloatList,
  Int64List,
  SequenceExample
}

import scala.collection.JavaConverters._

/**
  * Simple helpers for creating TensorFlow {@see Tensor}s of the right shapes.
  */
object TensorUtils {
  def replicate(encoding: Array[Float], length: Int): Array[Array[Float]] =
    Array.ofDim[Float](length, encoding.length).map(_ => encoding.clone())

  def create1Tensor(encoding: Array[Float]): Tensor[java.lang.Float] =
    Tensor.create(encoding, classOf[java.lang.Float])

  def create2Tensor(encoding: Array[Array[Float]]): Tensor[java.lang.Float] =
    Tensor.create(encoding, classOf[java.lang.Float])

  def buildIRSequenceExample(query: QueryContext,
                             docs: Array[Document],
                             numDocsPerQuery: Int): SequenceExample = {
    SequenceExample
      .newBuilder()
      .setContext(buildContextFeatures("query_text" -> query.queryString))
      .setFeatureLists(buildFeatureLists(docs, numDocsPerQuery))
      .build()
  }

  def transposeDocs(docs: Array[Document],
                    numDocsPerQuery: Int): Map[String, Array[Float]] = {
    case class FeatureVal(name: String, value: Float, docIdx: Int)
    docs
      .slice(0, math.min(docs.length, numDocsPerQuery))
      .zipWithIndex
      .flatMap {
        case (doc: Document, idx: Int) =>
          doc.numericFeatures.map {
            case (feature, value) => FeatureVal(feature, value, idx)
          }
      }
      .groupBy(_.name)
      .mapValues(_.sortBy(_.docIdx).map(_.value).padTo(numDocsPerQuery, 0f))
  }

  def buildContextFeatures(nodePairs: (String, String)*): Features = {
    nodePairs
      .foldLeft(Features.newBuilder()) {
        case (bldr, nodePair) =>
          bldr
            .putFeature(
              nodePair._1,
              Feature
                .newBuilder()
                .setBytesList(
                  BytesList
                    .newBuilder()
                    .addValue(
                      ByteString.copyFrom(nodePair._2.getBytes("UTF-8"))
                    )
                    .build()
                )
                .build()
            )
      }
      .build()
  }

  def buildFeatureLists(documents: Array[Document],
                        numDocsPerQuery: Int): FeatureLists = {
    transposeDocs(documents, numDocsPerQuery)
      .foldLeft(FeatureLists.newBuilder()) {
        case (bldr, (nodeName: String, featureValues: Array[Float])) =>
          bldr.putFeatureList(
            nodeName,
            buildSingleFeatureFloatList(
              featureValues.map(java.lang.Float.valueOf)
            )
          )
      }
      .putFeatureList(
        "pos",
        buildSingleFeatureIntList(Array.fill(numDocsPerQuery)(0L))
      )
      .build()
  }

  def toFeature(featureValues: Array[java.lang.Long]): Feature = {
    Feature
      .newBuilder()
      .setInt64List(
        Int64List.newBuilder().addAllValue(featureValues.toList.asJava)
      )
      .build()
  }

  def toFeature(featureValues: Array[java.lang.Float]): Feature = {
    Feature
      .newBuilder()
      .setFloatList(
        FloatList.newBuilder().addAllValue(featureValues.toList.asJava)
      )
      .build()
  }

  def buildSingleFeatureIntList(
    featureValues: Array[java.lang.Long]
  ): FeatureList = {
    FeatureList
      .newBuilder()
      .addFeature(toFeature(featureValues))
      .build()
  }

  def buildSingleFeatureFloatList(
    featureValues: Array[java.lang.Float]
  ): FeatureList = {
    FeatureList
      .newBuilder()
      .addFeature(toFeature(featureValues))
      .build()
  }

}
