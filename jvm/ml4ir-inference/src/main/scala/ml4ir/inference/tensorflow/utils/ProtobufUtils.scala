package ml4ir.inference.tensorflow.utils

import com.google.protobuf.ByteString
import ml4ir.inference.tensorflow.data.{Document, QueryContext}
import org.tensorflow.example._
import org.tensorflow.DataType

import scala.collection.JavaConverters._
import java.lang.{Float => JFloat, Long => JLong}
import scala.reflect.ClassTag

// TODO: these may not be necessary? Probably necessary to *construct* the QueryContext / Array[Document] actually
case class FeatureField(nodeName: String, dType: DataType)

case class FeatureConfig(contextFeatures: List[FeatureField] = List.empty,
                         documentFeatures: List[FeatureField] = List.empty,
                         numDocsPerQuery: Option[Int] = None,
                         queryLength: Option[Int] = None)

object FeatureConfig {
  // zero-arg constructor to be nice to Java
  def apply(): FeatureConfig =
    new FeatureConfig(List.empty, List.empty, None, None)
}

/**
  * Builder class for more easily instantiating SequenceExample protobufs from raw(-ish) features
  */
case class SequenceExampleBuilder(config: FeatureConfig = FeatureConfig()) {

  /**
    * Functional API allowing the builder to act like a function to transform query/documents into a scorable protobuf
    * @param query struct primarily containing the query text
    * @param docs array of document-feature structs
    * @return TensorFlow's protobuf structure containing the raw features in one SequenceExample packet
    */
  def apply(query: QueryContext, docs: Array[Document]): SequenceExample = {
    SequenceExample
      .newBuilder()
      .setContext(buildStringContextFeatures("query_text" -> query.queryString))
      .setFeatureLists(buildFeatureLists(docs))
      .build()
  }

  /**
    *
    * @param nodePairs
    * @return
    */
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

  /**
    *
    * @param documents
    * @return
    */
  def buildFeatureLists(documents: Array[Document]): FeatureLists = {
    val withFloats = transpose(documents.map(_.floatFeatures))
      .foldLeft(FeatureLists.newBuilder()) {
        case (bldr, (nodeName: String, featureValues: Array[Float])) =>
          bldr.putFeatureList(
            nodeName,
            floatList(featureValues.map(JFloat.valueOf))
          )
      }
    // hard code positions for now, but perhaps they should be dummy values at inference time?
    val positions: Array[Map[String, Long]] =
      documents.indices.map(i => Map("pos" -> i.toLong)).toArray
    val withFloatsAndInts =
      transpose(documents.map(_.int64Features) ++ positions)
        .foldLeft(withFloats) {
          case (bldr, (nodeName: String, featureValues: Array[Long])) =>
            bldr.putFeatureList(
              nodeName,
              longList(featureValues.map(JLong.valueOf))
            )
        }
    withFloatsAndInts.build()
  }

  /**
    * Effectively transforms an array of maps of features into a map of arrays of features: the "transpose" operation
    * @param docFeatures to have their features extracted out into one dense array per feature
    * @return map of feature-name -> padded dense vector of numeric features
    */
  def transpose[T: ClassTag](
    docFeatures: Array[Map[String, T]]
  ): Map[String, Array[T]] = {
    val numDocsPerQuery = config.numDocsPerQuery.getOrElse(docFeatures.length)
    case class FeatureVal(name: String, value: T, docIdx: Int)
    docFeatures
      .slice(0, math.min(docFeatures.length, numDocsPerQuery))
      .zipWithIndex
      .flatMap {
        case (doc: Map[String, T], idx: Int) =>
          doc.map {
            case (feature, value) => FeatureVal(feature, value, idx)
          }
      }
      .groupBy(_.name)
      .mapValues(_.sortBy(_.docIdx).map(_.value).toArray)
  }

  /**
    *
    * @param featureValues
    * @return
    */
  def toFeature(featureValues: Array[JLong]): Feature = {
    Feature
      .newBuilder()
      .setInt64List(
        Int64List.newBuilder().addAllValue(featureValues.toList.asJava)
      )
      .build()
  }

  /**
    *
    * @param featureValues
    * @return
    */
  def toFeature(featureValues: Array[JFloat]): Feature = {
    Feature
      .newBuilder()
      .setFloatList(
        FloatList.newBuilder().addAllValue(featureValues.toList.asJava)
      )
      .build()
  }

  /**
    *
    * @param featureValues
    * @return
    */
  def longList(featureValues: Array[JLong]): FeatureList = {
    FeatureList
      .newBuilder()
      .addFeature(toFeature(featureValues))
      .build()
  }

  /**
    *
    * @param featureValues
    * @return
    */
  def floatList(featureValues: Array[JFloat]): FeatureList = {
    FeatureList
      .newBuilder()
      .addFeature(toFeature(featureValues))
      .build()
  }

}
