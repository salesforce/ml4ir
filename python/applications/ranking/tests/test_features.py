# type: ignore
# TODO: Fix typing

from ml4ir.tests.test_base import RankingTestBase
from ml4ir.features import preprocessing
from ml4ir.features.feature_layer import get_sequence_encoding
import tensorflow as tf
import string
import numpy as np
from ml4ir.config.keys import TFRecordTypeKey


class RankingModelTest(RankingTestBase):
    def test_text_preprocesing(self):
        """
        Unit test text preprocessing
        """
        preprocessing_info = {"to_lower": True, "max_length": 20, "remove_punctuation": True}
        input_text = "ABCabc123!@#"
        processed_bytes_tensor = preprocessing.preprocess_text(input_text, preprocessing_info)
        processed_text = (
            tf.strings.unicode_encode(
                tf.cast(processed_bytes_tensor, tf.int32), output_encoding="UTF-8",
            )
            .numpy()
            .decode("utf-8")
        )

        # Clipping and padding to the maximum length set
        assert len(processed_bytes_tensor) == preprocessing_info["max_length"]

        # Converting to lower case
        assert processed_text.lower() == processed_text

        # Removing punctuation
        assert (
            processed_text.translate(str.maketrans("", "", string.punctuation)) == processed_text
        )

        # Overall
        assert processed_text.replace("\x00", "") == input_text.lower().translate(
            str.maketrans("", "", string.punctuation)
        )
        assert processed_text.replace("\x00", "") == "abcabc123"

    def test_sequence_encoding(self):
        """
        Unit test sequence embedding

        """
        batch_size = 50
        max_length = 20
        embedding_size = 128
        encoding_size = 512
        feature_info = {
            "feature_layer_info": {
                "embedding_size": embedding_size,
                "encoding_type": "bilstm",
                "encoding_size": encoding_size,
            },
            "preprocessing_info": {"max_length": max_length},
            "tfrecord_type": TFRecordTypeKey.CONTEXT,
        }

        """
        Input sequence tensor should be of type integer
        If float, it will be cast to uint8 as we use this to
        create one-hot representation of each time step

        If sequence tensor is a context feature, the shape can be either
        [batch_size, max_length] or [batch_size, 1, max_length]
        sand the method will tile the output embedding for all records.
        """
        sequence_tensor = np.random.randint(256, size=(batch_size, 1, max_length))

        sequence_encoding = get_sequence_encoding(sequence_tensor, feature_info)

        assert sequence_encoding.shape[0] == batch_size
        assert (
            sequence_encoding.shape[1] == 1
            if feature_info["tfrecord_type"] == TFRecordTypeKey.CONTEXT
            else self.args.max_num_records
        )
        assert sequence_encoding.shape[2] == encoding_size
