import numpy as np
from scipy import stats
import unittest
import warnings
from ml4ir.base.model.relevance_model import perform_click_rank_dist_paired_t_test, compute_stats_from_stream

warnings.filterwarnings("ignore")


class TestTtestCalculation(unittest.TestCase):
    """
    Testing the calculation of paired t-test statistic and its p-value.
    """

    def t_test_calculation(self, a, b, batchzs, pvalue_threshold):
        n = len(a)
        expected_t_test_stat, expected_pvalue = stats.ttest_rel(a, b)
        expected_decision = expected_pvalue < pvalue_threshold
        agg_mean, agg_count, agg_M2 = 0, 0, 0
        for bcount in range(int(n/batchzs)):
            batch_a = a[bcount*batchzs : (bcount+1)*(batchzs)]
            batch_b = b[bcount*batchzs : (bcount+1)*(batchzs)]
            d = batch_a - batch_b
            agg_count, agg_mean, agg_M2 = compute_stats_from_stream(d, agg_count, agg_mean, agg_M2)

        t_test_stat, pvalue = perform_click_rank_dist_paired_t_test(agg_mean, np.sqrt(agg_M2/agg_count), agg_count)
        decision = pvalue < pvalue_threshold
        assert decision == expected_decision
        #assert np.isclose(expected_pvalue, pvalue, atol=0.001)

    def test_1(self):
        pvalue_threshold = 0.1
        n = 100
        batchzs = 10
        a = np.random.randn(n) + 1
        b = np.random.randn(n)
        self.t_test_calculation(a, b, batchzs, pvalue_threshold)

    def test_2(self):
        pvalue_threshold = 0.1
        n = 10
        batchzs = 2
        a = np.random.randn(n) + 20
        b = np.random.randn(n)
        self.t_test_calculation(a, b, batchzs, pvalue_threshold)

    def test_3(self):
        pvalue_threshold = 0.1
        n = 50
        batchzs = 5
        a = np.random.randn(n)
        b = np.array(a)
        b[0] += 1
        self.t_test_calculation(a, b, batchzs, pvalue_threshold)

    def test_4(self):
        pvalue_threshold = 0.05
        n = 50
        batchzs = 5
        a = np.random.randn(n)
        b = np.array(a)
        b[0] += 1
        self.t_test_calculation(a, b, batchzs, pvalue_threshold)

    def test_5(self):
        pvalue_threshold = 0.05
        n = 100
        batchzs = 5
        a = np.random.randn(n)
        b = np.random.randn(n)
        self.t_test_calculation(a, b, batchzs, pvalue_threshold)



if __name__ == "__main__":
    unittest.main()
