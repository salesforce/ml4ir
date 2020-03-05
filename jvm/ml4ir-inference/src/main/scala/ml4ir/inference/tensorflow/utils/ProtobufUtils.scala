package ml4ir.inference.tensorflow.utils

import com.google.protobuf.ByteString
import ml4ir.inference.tensorflow.data.{Document, QueryContext}
import org.tensorflow.example._
import org.tensorflow.DataType

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

// TODO: these may not be necessary? Probably necessary to *construct* the QueryContext / Array[Document] actually
case class FeatureField(nodeName: String, dType: DataType)

case class FeatureConfig(contextFeatures: List[FeatureField] = List.empty,
                         documentFeatures: List[FeatureField] = List.empty,
                         numDocsPerQuery: Option[Int] = None,
                         queryLength: Option[Int] = None)

object FeatureConfig {
  def apply(): FeatureConfig =
    new FeatureConfig(List.empty, List.empty, None, None)
}

/**
  *
  */
case class SequenceExampleBuilder(config: FeatureConfig = FeatureConfig()) {

  def apply(query: QueryContext, docs: Array[Document]): SequenceExample = {
    SequenceExample
      .newBuilder()
      .setContext(buildStringContextFeatures("query_text" -> query.queryString))
      .setFeatureLists(buildFeatureLists(docs))
      .build()
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

  def buildFeatureLists(documents: Array[Document]): FeatureLists = {
    val withFloats = transpose(documents.map(_.floatFeatures))
      .foldLeft(FeatureLists.newBuilder()) {
        case (bldr, (nodeName: String, featureValues: Array[Float])) =>
          bldr.putFeatureList(
            nodeName,
            buildSingleFeatureFloatList(
              featureValues.map(java.lang.Float.valueOf)
            )
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
              buildSingleFeatureIntList(
                featureValues.map(java.lang.Long.valueOf)
              )
            )
        }
    withFloatsAndInts.build()
  }

  /**
    * Effectively transforms an array of maps of features into a map of arrays of features: the "transpose" operation
    * @param docs to have their features extracted out into one dense array per feature
    *             (note: currently Document only has float features)
    * @return map of feature-name -> padded dense vector of numeric features
    */
  def transposeDocs(docs: Array[Document]): Map[String, Array[Float]] = {
    case class FeatureVal(name: String, value: Float, docIdx: Int)
    val numDocsPerQuery = config.numDocsPerQuery.getOrElse(docs.length)
    docs
      .slice(0, math.min(docs.length, numDocsPerQuery))
      .zipWithIndex
      .flatMap {
        case (doc: Document, idx: Int) =>
          doc.floatFeatures.map {
            case (feature, value) => FeatureVal(feature, value, idx)
          }
      }
      .groupBy(_.name)
      .mapValues(_.sortBy(_.docIdx).map(_.value).padTo(numDocsPerQuery, 0f))
  }

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
