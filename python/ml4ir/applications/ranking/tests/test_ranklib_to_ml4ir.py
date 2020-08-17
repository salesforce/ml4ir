import unittest
import os
import warnings
from ml4ir.base.data import ranklib_to_ml4ir
import pandas as pd


warnings.filterwarnings("ignore")

INPUT_FILE = "ml4ir/applications/ranking/tests/data/sample.txt"
OUTPUT_FILE = "ml4ir/applications/ranking/tests/data/sample_ml4ir.csv"
QUERY_ID_NAME = 'qid'
RELEVANCE_NAME = 'relevance'
KEEP_ADDITIONAL_INFO = 1
GL_2_CLICKS = 1

class Test_ranklib_to_ml4ir(unittest.TestCase):
    def setUp(self):
        ranklib_to_ml4ir.convert(INPUT_FILE, OUTPUT_FILE, KEEP_ADDITIONAL_INFO, GL_2_CLICKS)

    def test(self):
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
