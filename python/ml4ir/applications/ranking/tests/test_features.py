import string
import numpy as np

from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.features import preprocessing
from ml4ir.base.features.feature_layer import get_sequence_encoding
from ml4ir.base.config.keys import SequenceExampleTypeKey


class RankingModelTest(RankingTestBase):
    def test_text_preprocesing(self):
        """
        Unit test text preprocessing
        """
        input_text = "ABCabc123!@#"
        processed_text = (
            preprocessing.preprocess_text(input_text, remove_punctuation=True, to_lower=True)
            .numpy()
            .decode("utf-8")
        )

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
                "type": "numeric",
                "fn": "get_sequence_encoding",
                "args": {
                    "embedding_size": embedding_size,
                    "encoding_type": "bilstm",
                    "encoding_size": encoding_size,
                    "max_length": max_length,
                },
            },
            "tfrecord_type": SequenceExampleTypeKey.CONTEXT,
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
            if feature_info["tfrecord_type"] == SequenceExampleTypeKey.CONTEXT
            else self.args.max_num_records
        )
        assert sequence_encoding.shape[2] == encoding_size
