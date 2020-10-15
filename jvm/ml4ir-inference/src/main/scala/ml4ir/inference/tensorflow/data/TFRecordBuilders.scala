package ml4ir.inference.tensorflow.data

import com.google.common.base.Charsets
import com.google.protobuf.ByteString
import org.tensorflow.example._
import java.lang.{Float => JFloat, Long => JLong}

import scala.collection.JavaConverters._
import scala.reflect.ClassTag

/**
  * For converting a concrete class of type T into an {@see Example} TFRecord to perform inference
  * @param featuresPreprocessor for any JVM-only preprocessing on the input before protobuf construction
  * @tparam T only constrained by the type of the featuresPreprocessor supplied
  */
class ExampleBuilder[T](featuresPreprocessor: FeaturePreprocessor[T]) extends TFRecordBuilderUtils {
  def apply(rawInput: T): Example = {
    val example: MultiFeatures = featuresPreprocessor(rawInput)
    val features = buildMultiFeatures(example)
    Example
      .newBuilder()
      .setFeatures(features)
      .build()
  }
}

/**
  * For converting a concrete class of type T into an {@see SequenceExample} TFRecord to perform inference
  * @param contextFeaturesPreprocessor for any JVM-only preprocessing on the input context before protobuf construction
  * @param sequenceFeaturesPreprocessor for any JVM-only preprocessing on the input sequence records before protobuf
  *                                     construction
  * @tparam C whatever type the contextFeaturesPreprocessor knows how to preprocess
  * @tparam S whatever type the sequenceFeaturesPreprocessor knows how to preprocess
  */
class SequenceExampleBuilder[C, S](contextFeaturesPreprocessor: FeaturePreprocessor[C],
                                   sequenceFeaturesPreprocessor: FeaturePreprocessor[S])
    extends TFRecordBuilderUtils {

  def apply(context: C, sequence: List[S]): SequenceExample =
    fromMultiFeatures(contextFeaturesPreprocessor(context), sequence.map(sequenceFeaturesPreprocessor).toArray)

  /**
    * Java-friendly API
    * @param context struct primarily containing e.g. query text, userId or cookie, other document-indepenent features
    * @param sequence list of document-feature structs
    * @return TensorFlow's protobuf structure containing the raw features in one SequenceExample packet
    */
  def build(context: C, sequence: java.util.List[S]): SequenceExample = apply(context, sequence.asScala.toList)

  /**
    * Functional API allowing the builder to act like a function to transform query/documents into a scorable protobuf
    * @param context struct primarily containing e.g. query text, userId or cookie, other document-indepenent features
    * @param docs array of document-feature structs
    * @return TensorFlow's protobuf structure containing the raw features in one SequenceExample packet
    */
  def fromMultiFeatures(context: MultiFeatures, docs: Array[MultiFeatures]): SequenceExample = {
    val contextFeatures: Features = buildMultiFeatures(context)
    val docFeatures: FeatureLists = buildMultiFeatureLists(docs)
    SequenceExample
      .newBuilder()
      .setContext(contextFeatures)
      .setFeatureLists(docFeatures)
      .build()
  }
}

/**
  * Functionality shared between both {@see Example} and {@see SequenceExample} instantiation.  Primarily just packing
  * primitives from the {@code MultiFeatures} into {@see Features} and {@see FeatureLists} protobuf structures
  */
trait TFRecordBuilderUtils {

  protected def buildMultiFeatures(features: MultiFeatures): Features = {
    val withStringFeatures = features.stringFeatures
      .foldLeft(Features.newBuilder()) {
        case (builder, (nodeName: String, stringFeature: String)) =>
          builder.putFeature(nodeName, toFeature(stringFeature))
      }
    val withStringAndFloatFeatures = features.floatFeatures
      .foldLeft(withStringFeatures) {
        case (builder, (nodeName: String, floatFeature: Float)) =>
          builder.putFeature(nodeName, toFeature(floatFeature))
      }
    val withFloatsAndIntsAndStrings = features.int64Features
      .foldLeft(withStringAndFloatFeatures) {
        case (builder, (nodeName: String, longFeature: Long)) =>
          builder.putFeature(nodeName, toFeature(longFeature))
      }
    withFloatsAndIntsAndStrings.build()
  }

  protected def buildMultiFeatureLists(features: Array[MultiFeatures]): FeatureLists = {
    val withFloats = transpose(features.map(_.floatFeatures))
      .foldLeft(FeatureLists.newBuilder()) {
        case (builder, (name: String, featureValues: Array[Float])) =>
          builder.putFeatureList(name, floats(featureValues.map(JFloat.valueOf)))
      }
    val withFloatsAndInts =
      transpose(features.map(_.int64Features))
        .foldLeft(withFloats) {
          case (builder, (name: String, featureValues: Array[Long])) =>
            builder.putFeatureList(name, longs(featureValues.map(JLong.valueOf)))
        }
    val withFloatsAndIntsAndStrings =
      transpose(features.map(_.stringFeatures))
        .foldLeft(withFloatsAndInts) {
          case (builder, (name: String, featureValues: Array[String])) =>
            builder.putFeatureList(name, strings(featureValues))
        }
    withFloatsAndIntsAndStrings.build()
  }

  /**
    * Effectively transforms an array of maps of features into a map of arrays of features: the "transpose" operation
    * @param docFeatures to have their features extracted out into one dense array per feature
    * @return map of feature-name -> dense vector of numeric features
    */
  protected def transpose[T: ClassTag](
      docFeatures: Array[Map[String, T]]
  ): Map[String, Array[T]] = {
    case class FeatureVal(name: String, value: T, docIdx: Int)
    val featureSet: Set[String] = docFeatures.map(_.keySet).reduce(_ union _)
    docFeatures
      .slice(0, docFeatures.length)
      .zipWithIndex
      .flatMap {
        case (doc: Map[String, T], idx: Int) =>
          featureSet.map(name => FeatureVal(name, doc(name), idx))
      }
      .groupBy(_.name)
      .mapValues(_.sortBy(_.docIdx).map(_.value).toArray)
  }

  /**
    *
    * @param featureValues
    * @return
    */
  protected def toFeature(featureValues: Array[JLong]): Feature = {
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
  protected def toFeature(featureValues: Array[JFloat]): Feature = {
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
  protected def toFeature(featureValues: Array[String]): Feature = {
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
  protected def longs(featureValues: Array[JLong]): FeatureList = {
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
  protected def floats(featureValues: Array[JFloat]): FeatureList = {
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
  protected def strings(featureValues: Array[String]): FeatureList = {
    FeatureList
      .newBuilder()
      .addFeature(toFeature(featureValues))
      .build()
  }

  protected def toFeature(stringFeature: String): Feature = {
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
  protected def toFeature(floatFeature: Float): Feature = {
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
  protected def toFeature(longFeature: Long): Feature = {
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
