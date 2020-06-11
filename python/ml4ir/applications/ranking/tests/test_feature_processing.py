import string

from ml4ir.applications.ranking.tests.test_base import RankingTestBase
from ml4ir.base.features import preprocessing


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
