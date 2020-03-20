from ml4ir.tests.test_base import RankingTestBase
from ml4ir.features import preprocessing
from ml4ir.features.feature_layer import _get_sequence_embedding
import tensorflow as tf
import string


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

    def test_sequence_embedding(self):
        """
        Unit test sequence embedding

        TODO
        """
        pass
