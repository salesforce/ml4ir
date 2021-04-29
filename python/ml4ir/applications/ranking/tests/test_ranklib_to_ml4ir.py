import unittest
import os
import warnings
from ml4ir.base.data import ranklib_helper
import pandas as pd


warnings.filterwarnings("ignore")

INPUT_FILE = "ml4ir/applications/ranking/tests/data/ranklib/train/sample.txt"
OUTPUT_FILE = "ml4ir/applications/ranking/tests/data/ranklib/train/sample_ml4ir.csv"
QUERY_ID_NAME = 'qid'
RELEVANCE_NAME = 'relevance'
KEEP_ADDITIONAL_INFO = 1
GL_2_CLICKS = 1
NON_ZERO_FEATURES_ONLY = 0


class TestRanklibConversion(unittest.TestCase):
    def setUp(self):
        pass

    def test_conversion(self):
        """Convert ranklib dataset to a csv"""

        ranklib_helper.ranklib_to_csv(INPUT_FILE, OUTPUT_FILE, KEEP_ADDITIONAL_INFO,
                                      GL_2_CLICKS, NON_ZERO_FEATURES_ONLY, QUERY_ID_NAME, RELEVANCE_NAME)
        df = pd.read_csv(OUTPUT_FILE)
        assert QUERY_ID_NAME in df.columns and RELEVANCE_NAME in df.columns
        assert df[QUERY_ID_NAME].nunique() == 49
        if KEEP_ADDITIONAL_INFO == 1:
            assert len(df.columns) >= 138
        else:
            assert len(df.columns) == 138

        if GL_2_CLICKS == 1:
            assert sorted(list(df[RELEVANCE_NAME].unique())) == [0, 1]

    def tearDown(self):
        # Delete output file
        os.remove(OUTPUT_FILE)


if __name__ == "__main__":
    unittest.main()
