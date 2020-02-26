package ml4ir.inference.tensorflow.utils

import com.google.protobuf.ByteString
import ml4ir.inference.tensorflow.{Document, QueryContext}
import org.tensorflow.example._

import scala.collection.JavaConverters._

/**
  * TODO: this should be a config-driven builder class:
  * val protoBuilder = IRSequenceExampleBuilder(config)
  * val proto: SequenceExample = protoBuilder.build(query, docs)
  */
object ProtobufUtils {

  def buildIRSequenceExample(query: QueryContext,
                             docs: Array[Document],
                             numDocsPerQuery: Int): SequenceExample = {
    SequenceExample
      .newBuilder()
      .setContext(buildStringContextFeatures("query_text" -> query.queryString))
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

  def buildStringContextFeatures(nodePairs: (String, String)*): Features = {
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
      } // manually putting a dummy value for the position feature
      .putFeatureList(
        "pos",
        buildSingleFeatureIntList(Array.fill(numDocsPerQuery)(-1L))
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
