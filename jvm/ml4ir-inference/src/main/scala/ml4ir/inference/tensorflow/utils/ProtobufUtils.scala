package ml4ir.inference.tensorflow.utils

import com.google.protobuf.ByteString
import ml4ir.inference.tensorflow.data.{Example, MultiFeatures}
import org.tensorflow.example._
import org.tensorflow.DataType

import scala.collection.JavaConverters._
import java.lang.{Float => JFloat, Long => JLong}

import com.google.common.base.Charsets

import scala.reflect.ClassTag

//// TODO: these may not be necessary? Probably necessary to *construct* the QueryContext / Array[Document] actually
//case class FeatureField(servingName: String, nodeName: String, dType: DataType, defaultValue: String)
//
//case class FeatureConfig(contextFeatures: List[FeatureField] = List.empty,
//                         documentFeatures: List[FeatureField] = List.empty,
//                         numDocsPerQuery: Option[Int] = None,
//                         queryLength: Option[Int] = None)
//
//object FeatureConfig {
//  // zero-arg constructor to be nice to Java
//  def apply(): FeatureConfig =
//    new FeatureConfig(List.empty, List.empty, None, None)
//  def apply(contextFeatures: java.util.List[FeatureField], documentFeatures: java.util.List[FeatureField]) = {
//    new FeatureConfig(
//      contextFeatures.asScala.toList,
//      documentFeatures.asScala.toList,
//      None,
//      None
//    )
//  }
//}

/**
  * Builder class for more easily instantiating SequenceExample protobufs from raw(-ish) features
  */
case class SequenceExampleBuilder() {

  /**
    * Functional API allowing the builder to act like a function to transform query/documents into a scorable protobuf
    * @param context struct primarily containing the query text
    * @param docs array of document-feature structs
    * @return TensorFlow's protobuf structure containing the raw features in one SequenceExample packet
    */
  def apply(context: Example, docs: Array[Example]): SequenceExample = {
    val contextFeatures: Features = buildMultiFeatures(context.features)
    val docFeatures = buildMultiFeatureLists(docs.map(_.features))
    SequenceExample
      .newBuilder()
      .setContext(contextFeatures)
      .setFeatureLists(docFeatures)
      .build()
  }

  def buildMultiFeatures(features: MultiFeatures): Features = {
    val withStringFeatures = features.stringFeatures
      .foldLeft(Features.newBuilder()) {
        case (bldr, (nodeName: String, stringFeature: String)) =>
          bldr.putFeature(nodeName, toFeature(stringFeature))
      }
    val withStringAndFloatFeatures = features.floatFeatures
      .foldLeft(withStringFeatures) {
        case (bldr, (nodeName: String, floatFeature: Float)) =>
          bldr.putFeature(nodeName, toFeature(floatFeature))
      }
    val withFloatsAndIntsAndStrings = features.int64Features
      .foldLeft(withStringAndFloatFeatures) {
        case (bldr, (nodeName: String, longFeature: Long)) =>
          bldr.putFeature(nodeName, toFeature(longFeature))
      }
    withFloatsAndIntsAndStrings.build()
  }

  def buildMultiFeatureLists(features: Array[MultiFeatures]): FeatureLists = {
    val withFloats = transpose(features.map(_.floatFeatures))
      .foldLeft(FeatureLists.newBuilder()) {
        case (bldr, (name: String, featureValues: Array[Float])) =>
          bldr.putFeatureList(name, floats(featureValues.map(JFloat.valueOf)))
      }
    val withFloatsAndInts =
      transpose(features.map(_.int64Features))
        .foldLeft(withFloats) {
          case (bldr, (name: String, featureValues: Array[Long])) =>
            bldr.putFeatureList(name, longs(featureValues.map(JLong.valueOf)))
        }
    val withFloatsAndIntsAndStrings =
      transpose(features.map(_.stringFeatures))
        .foldLeft(withFloatsAndInts) {
          case (bldr, (name: String, featureValues: Array[String])) =>
            bldr.putFeatureList(name, strings(featureValues))
        }
    withFloatsAndIntsAndStrings.build()
  }

  /**
    * Effectively transforms an array of maps of features into a map of arrays of features: the "transpose" operation
    * @param docFeatures to have their features extracted out into one dense array per feature
    * @return map of feature-name -> padded dense vector of numeric features
    */
  def transpose[T: ClassTag](
      docFeatures: Array[Map[String, T]]
  ): Map[String, Array[T]] = {
    // val numDocsPerQuery = config.numDocsPerQuery.getOrElse(docFeatures.length)
    case class FeatureVal(name: String, value: T, docIdx: Int)
    val featureSet: Set[String] = docFeatures.map(_.keySet).reduce(_ union _)
    docFeatures
      .slice(0, docFeatures.length) // math.min(docFeatures.length, numDocsPerQuery))
      .zipWithIndex
      .flatMap {
        case (doc: Map[String, T], idx: Int) =>
          featureSet.map(name => FeatureVal(name, doc(name), idx))
        /*doc.map {
            case (feature, value) => FeatureVal(feature, value, idx)
          }*/
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
  def toFeature(featureValues: Array[String]): Feature = {
    Feature
      .newBuilder()
      .setBytesList(
        BytesList
          .newBuilder()
          .addAllValue(
            featureValues.toList
              .map(a => ByteString.copyFrom(a.getBytes))
              .asJava
          )
          .build()
      )
      .build()
  }

  /**
    *
    * @param featureValues
    * @return
    */
  def longs(featureValues: Array[JLong]): FeatureList = {
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
  def floats(featureValues: Array[JFloat]): FeatureList = {
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
  def strings(featureValues: Array[String]): FeatureList = {
    FeatureList
      .newBuilder()
      .addFeature(toFeature(featureValues))
      .build()
  }

  def toFeature(stringFeature: String): Feature = {
    Feature
      .newBuilder()
      .setBytesList(
        BytesList
          .newBuilder()
          .addValue(ByteString.copyFrom(stringFeature.getBytes(Charsets.UTF_8)))
          .build()
      )
      .build()
  }

  /**
    *
    * @param floatFeature
    * @return
    */
  def toFeature(floatFeature: Float): Feature = {
    Feature
      .newBuilder()
      .setFloatList(
        FloatList
          .newBuilder()
          .addValue(floatFeature)
          .build()
      )
      .build()
  }

  /**
    *
    * @param longFeature
    * @return
    */
  def toFeature(longFeature: Long): Feature = {
    Feature
      .newBuilder()
      .setInt64List(
        Int64List
          .newBuilder()
          .addValue(longFeature)
          .build()
      )
      .build()
  }
}
