import numpy as np
from scipy import stats
import unittest
import warnings
from ml4ir.applications.ranking.t_test import perform_click_rank_dist_paired_t_test, compute_stats_from_stream

warnings.filterwarnings("ignore")

np.random.seed(123)


class TestTtestCalculation(unittest.TestCase):
    """
    Testing the calculation of paired t-test statistic and its p-value.
    """

    def t_test_calculation(self, a_bucket, b_bucket, batch_size):
        n = len(a_bucket)
        expected_t_test_stat, expected_pvalue = stats.ttest_rel(a_bucket, b_bucket)
        agg_mean, agg_count, agg_M2 = 0, 0, 0
        for bcount in range(int(n/batch_size)):
            batch_a = a_bucket[bcount*batch_size : (bcount+1)*(batch_size)]
            batch_b = b_bucket[bcount*batch_size : (bcount+1)*(batch_size)]
            d = batch_a - batch_b
            agg_count, agg_mean, agg_M2 = compute_stats_from_stream(d, agg_count, agg_mean, agg_M2)

        t_test_stat, pvalue = perform_click_rank_dist_paired_t_test(agg_mean, (agg_M2/(agg_count-1)), agg_count)
        assert np.isclose(expected_pvalue, pvalue, atol=0.0001)

    def test_1(self):
        n = 100
        batch_size = 10
        batch_a = np.random.randn(n) + 1
        batch_b = np.random.randn(n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_2(self):
        n = 10
        batch_size = 3
        batch_a = np.random.randn(n) + 20
        batch_b = np.random.randn(n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_3(self):
        n = 50
        batch_size = 5
        batch_a = np.random.randn(n)
        batch_b = np.array(batch_a)
        batch_b[0] += 1
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_4(self):
        n = 50
        batch_size = 5
        batch_a = np.random.randn(n)
        batch_b = np.array(batch_a)
        batch_b[0] += 1
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_5(self):
        n = 100
        batch_size = 5
        batch_a = np.random.randn(n)
        batch_b = np.random.randn(n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_6(self):
        n = 100
        batch_size = 5
        batch_a = np.random.exponential(scale=0.1, size=n)
        batch_b = np.random.exponential(scale=0.5, size=n)
        self.t_test_calculation(batch_a, batch_b, batch_size)
        
        
    def test_6(self):
        n = 100
        batch_size = 5
        batch_a = np.random.exponential(scale=0.1, size=n)
        batch_b = np.random.exponential(scale=0.5, size=n)
        self.t_test_calculation(batch_a, batch_b, batch_size)


    def test_7(self):
        n = 10
        batch_size = 2
        batch_a = np.random.exponential(scale=1.0, size=n)
        batch_b = np.random.exponential(scale=0.2, size=n)
        self.t_test_calculation(batch_a, batch_b, batch_size)


    def test_7(self):
        n = 10
        batch_size = 2
        batch_a = np.random.exponential(scale=1.0, size=n)
        batch_b = np.random.exponential(scale=0.2, size=n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_8(self):
        n = 50
        batch_size = 10
        batch_a = np.random.exponential(scale=0.1, size=n)
        batch_b = np.random.exponential(scale=0.1, size=n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_8(self):
        n = 50
        batch_size = 10
        batch_a = np.random.exponential(scale=0.1, size=n)
        batch_b = np.random.exponential(scale=0.1, size=n)
        self.t_test_calculation(batch_a, batch_b, batch_size)

    def test_9(self):
        n = 50
        batch_size = 5
        batch_a = np.random.exponential(scale=0.1, size=n)
        batch_b = batch_a + 0.001
        self.t_test_calculation(batch_a, batch_b, batch_size)


if __name__ == "__main__":
    unittest.main()
