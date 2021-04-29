import string

from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.features import preprocessing

import tensorflow as tf
import numpy as np

class RankingModelTest(RankingTestBase):
    def test_text_preprocesing(self):
        """
        Asserts the preprocessing of a string tensor by
        converting it to its lower case form and removing punctuations
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

    def test_text_preprocesing_with_replace_by_whitespace(self):
        """
        Asserts the preprocessing of a string tensor with custom punctuation character and whitespace replacement character
        """
        input_text = " # abc. bcd-$#efg@hij ."
        processed_text = (
            preprocessing.preprocess_text(input_text,
                                          remove_punctuation=True,
                                          to_lower=True,
                                          punctuation=".-$#",
                                          replace_with_whitespace=True)
                .numpy()
                .decode("utf-8")
        )

        self.assertEqual("abc bcd efg@hij", processed_text)

    def test_get_one_hot_vectorizer(self):
        """
        Asserts ml4ir.base.features.preprocessing.get_one_hot_vectorizer
        """
        feature_info = {
            "name": "categorical_variable",
            "feature_layer_info": {
                "fn": "categorical_indicator_with_vocabulary_file",
                "args": {
                    "vocabulary_file": "ml4ir/applications/classification/tests/data/configs/vocabulary/entity_id.csv",
                    "num_oov_buckets": 1,
                },
            },
            "default_value": "",
        }

        one_hot_vectorizer = preprocessing.get_one_hot_label_vectorizer(feature_info, self.file_io)

        # Assert 1st position
        one_hot_labels = one_hot_vectorizer(tf.constant(["AAA"]))
        expected_one_hot_labels = tf.constant([[1.] + 8*[0.]])
        assert tf.reduce_all(tf.equal(one_hot_labels, expected_one_hot_labels))

        # Assert 7th position
        one_hot_labels = one_hot_vectorizer(tf.constant(["GGG"]))
        expected_one_hot_labels = tf.constant([6*[0.] + [1.] + 2*[0.]])
        assert tf.reduce_all(tf.equal(one_hot_labels, expected_one_hot_labels))

        # Assert last position for out of vocabulary
        one_hot_labels = one_hot_vectorizer(tf.constant(["out of vocabulary"]))
        expected_one_hot_labels = tf.constant([8*[0.] + [1.]])
        assert tf.reduce_all(tf.equal(one_hot_labels, expected_one_hot_labels))

    def test_split_and_pad_string(self):
        """
        Asserts ml4ir.base.features.preprocessing.split_and_pad_string
        """
        input_text = tf.constant(["_A_BC_ab_c1_23!@_#"])

        split_text = preprocessing.split_and_pad_string(input_text, split_char="_", max_length=25)
        expected_split_text = tf.constant(["", "A", "BC", "ab", "c1", "23!@", "#"] + [""]*18)

        assert tf.reduce_all(tf.equal(split_text, expected_split_text))

    def testing_click_conversion(self):
        typ = 'int'
        label_vector = np.ones(10, dtype=typ)
        label_vector = tf.convert_to_tensor(label_vector)
        clicks = preprocessing.convert_label_to_clicks(label_vector, typ)
        comp = tf.equal(label_vector, clicks)
        assert sum(tf.dtypes.cast(comp, 'int8')) == 10

        typ = 'float'
        label_vector = np.ones(10, dtype=typ)
        label_vector[0] = 5
        label_vector[-1] = 5
        label_vector = tf.convert_to_tensor(label_vector)
        clicks = preprocessing.convert_label_to_clicks(label_vector, typ)
        assert clicks[0] == 1 and clicks[-1] == 1 and sum(clicks[1:-1]) == 0

        typ = 'int'
        label_vector = np.zeros(10, dtype=typ)
        label_vector = tf.convert_to_tensor(label_vector)
        clicks = preprocessing.convert_label_to_clicks(label_vector, typ)
        assert sum(tf.dtypes.cast(comp, 'int8')) == 10
