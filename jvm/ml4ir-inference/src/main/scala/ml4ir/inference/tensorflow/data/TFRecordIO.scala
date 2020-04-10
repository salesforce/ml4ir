package ml4ir.inference.tensorflow.data
import java.io.{ByteArrayOutputStream, File, FileOutputStream}

import com.google.common.hash.Hashing
import com.google.common.io.LittleEndianDataOutputStream
import org.tensorflow.example.SequenceExample

object TFRecordIO {
  val kMaskDelta = 0xa282ead8

  def hash(in: Array[Byte]): Int = Hashing.crc32c().hashBytes(in).asInt()

  def mask(crc: Int): Int = ((crc >>> 15) | (crc << 17)) + kMaskDelta

  case class SerializedChunks(length: Array[Byte],
                              crcLength: Array[Byte],
                              exSerialized: Array[Byte],
                              crcEx: Array[Byte])

  def write(exSerialized: Array[Byte], fileName: String) = {
    val SerializedChunks(length, crcLength, _, crcEx) = prepare(exSerialized)

    val out = new FileOutputStream(new File(fileName))
    out.write(length)
    out.write(crcLength)
    out.write(exSerialized)
    out.write(crcEx)
    out.close()
  }

  def write(seqExamples: List[SequenceExample], fileName: String) = {
    seqExamples
      .map(_.toByteArray)
      .map(prepare)
      .foldLeft(new FileOutputStream(fileName)) {
        case (out, SerializedChunks(length, crcLength, exSerialized, crcEx)) =>
          out.write(length)
          out.write(crcLength)
          out.write(exSerialized)
          out.write(crcEx)
          // FIXME: newline between each example?!? Not sure.
          out.write("\n".getBytes)
          out
      }
      .close()
  }

  def prepare(exSerialized: Array[Byte]) = {
    val length: Array[Byte] = LittleEndianEncoding.encodeLong(exSerialized.length)
    val crcLength: Array[Byte] = LittleEndianEncoding.encodeInt(mask(hash(length)))
    val crcEx: Array[Byte] = LittleEndianEncoding.encodeInt(mask(hash(exSerialized)))
    SerializedChunks(length, crcLength, exSerialized, crcEx)
  }
}

object LittleEndianEncoding {
  def encodeLong(in: Long): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    val out = new LittleEndianDataOutputStream(baos)
    out.writeLong(in)
    baos.toByteArray
  }

  def encodeInt(in: Int): Array[Byte] = {
    val baos = new ByteArrayOutputStream()
    val out = new LittleEndianDataOutputStream(baos)
    out.writeInt(in)
    baos.toByteArray
  }
}
